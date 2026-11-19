from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .branch.graph_facts import branch_miss_shape_key
from .branch.functional_groups import _formyl_prefix_allowed
from .candidate_facts import PrefixCandidate
from .constants import HALO, MULT
from .formatting import format_prefixes as _format_prefixes
from .molecule_cache import mol_cache_dict, mol_cache_get
from .simple_prefixes import simple_external_prefix_candidates
from .stereochemistry import assigned_atom_stereochemistry


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
    *,
    assume_simple_external_prefix_miss: bool = False,
) -> tuple[PrefixCandidate, ...]:
    """Return structured branch-prefix facts before legacy string rendering."""

    cache = mol_cache_dict(ctx.mol, "branch_render_candidate_cache")
    key = (idx, parent_idx)
    candidates = cache.get(key)
    if candidates is None:
        candidates = (
            ()
            if assume_simple_external_prefix_miss
            else simple_external_prefix_candidates(ctx.mol, idx, parent_idx)
        )
        if not candidates:
            candidate = _local_functional_prefix_candidate(ctx.mol, idx, parent_idx, ctx.Chem)
            candidates = (candidate,) if candidate is not None else ()
        cache[key] = candidates
    if ctx.telemetry is not None:
        if candidates:
            for candidate in candidates:
                ctx.telemetry.branch_structured_prefix_candidates[_structured_prefix_key(candidate)] += 1
        else:
            miss_key = (
                branch_miss_shape_key(ctx.mol, idx, parent_idx)
                if ctx.telemetry.branch_miss_shape_detail
                else "simple_external_prefix"
            )
            ctx.telemetry.branch_structured_prefix_misses[miss_key] += 1
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


def _local_functional_prefix_candidate(mol, idx: int, parent_idx: int, Chem=None) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "H":
        return _isotopic_hydrogen_prefix(atom, idx, parent_idx)
    if symbol == "B":
        return (
            _pinacol_boronate_prefix(mol, idx, parent_idx)
            or _borono_prefix(mol, idx, parent_idx)
            or _dihaloboryl_prefix(mol, idx, parent_idx)
        )
    if symbol == "O":
        return _tbdms_silanoxy_prefix(mol, idx, parent_idx)
    if symbol == "P":
        return _structured_phosphoryl_prefix(mol, idx, parent_idx) or _bis_oxidanidyl_phosphanyl_prefix(
            mol,
            idx,
            parent_idx,
        )
    if atom.IsInRing():
        return _simple_ring_prefix(mol, idx, parent_idx)
    if symbol == "C":
        return (
            _sem_methyl_prefix(mol, idx, parent_idx)
            or _cyano_prefix(mol, idx, parent_idx)
            or _formyl_prefix(mol, idx, parent_idx)
            or _oxo_methyl_prefix(mol, idx, parent_idx)
            or _carboxylato_prefix(mol, idx, parent_idx)
            or _azabicyclo221_heptane_carbonyl_prefix(mol, idx, parent_idx, Chem)
            or _retained_acyl_prefix(mol, idx, parent_idx)
            or _carbamyl_prefix(mol, idx, parent_idx)
            or _retained_alkoxycarbonyl_prefix(mol, idx, parent_idx)
            or _carboxy_prefix(mol, idx, parent_idx)
        )
    if symbol == "N":
        return (
            _azido_prefix(mol, idx, parent_idx)
            or _nitro_prefix(mol, idx, parent_idx)
            or _sulfinylamino_prefix(mol, idx, parent_idx)
            or _mesylamino_prefix(mol, idx, parent_idx)
            or _sulfonylamino_prefix(mol, idx, parent_idx)
            or _ethenylcarbonyl_amino_prefix(mol, idx, parent_idx)
            or _alkoxycarbonyl_amino_prefix(mol, idx, parent_idx)
            or _trifluoroacetamido_prefix(mol, idx, parent_idx)
            or _retained_amido_prefix(mol, idx, parent_idx)
            or _bocamino_prefix(mol, idx, parent_idx)
            or _long_bocamino_prefix(mol, idx, parent_idx)
        )
    if symbol == "S":
        return (
            _aminosulfonyl_prefix(mol, idx, parent_idx)
            or _simple_sulfur_prefix(mol, idx, parent_idx)
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


def _sem_methyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygen_idx = _single_nonring_heavy_neighbor_with_symbol(mol, idx, parent_idx, "O")
    if oxygen_idx is None:
        return None
    first_carbon = _single_nonring_heavy_neighbor_with_symbol(mol, oxygen_idx, idx, "C")
    if first_carbon is None:
        return None
    second_carbon = _single_nonring_heavy_neighbor_with_symbol(mol, first_carbon, oxygen_idx, "C")
    if second_carbon is None:
        return None
    silicon_idx = _single_nonring_heavy_neighbor_with_symbol(mol, second_carbon, first_carbon, "Si")
    if silicon_idx is None:
        return None
    silicon_atoms = _trimethylsilyl_atoms(mol, silicon_idx, second_carbon)
    if silicon_atoms is None:
        return None
    return _prefix_candidate(
        "2-tms-ethoxymethyl",
        {idx, oxygen_idx, first_carbon, second_carbon, *silicon_atoms},
        idx,
        parent_idx,
        "sem-methyl",
    )


def _single_nonring_heavy_neighbor_with_symbol(
    mol,
    atom_idx: int,
    blocked_idx: int,
    symbol: str,
) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
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
    if (
        neighbor.GetSymbol() != symbol
        or neighbor.IsInRing()
        or bond is None
        or bond.GetBondTypeAsDouble() != 1.0
    ):
        return None
    return neighbor_idx


def _trimethylsilyl_atoms(mol, silicon_idx: int, blocked_idx: int) -> set[int] | None:
    silicon = mol.GetAtomWithIdx(silicon_idx)
    if silicon.GetSymbol() != "Si" or silicon.GetFormalCharge() != 0 or silicon.IsInRing() or silicon.GetIsotope():
        return None
    methyls = [
        nbr.GetIdx()
        for nbr in silicon.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetSymbol() == "C"
    ]
    if len(methyls) != 3:
        return None
    if not all(_is_plain_terminal_methyl(mol, methyl_idx, silicon_idx) for methyl_idx in methyls):
        return None
    return {silicon_idx, *methyls}


def _tbdms_silanoxy_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "O" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    silicon_idx = _single_nonring_heavy_neighbor_with_symbol(mol, idx, parent_idx, "Si")
    if silicon_idx is None:
        return None
    silicon_atoms = _tbdms_atoms(mol, silicon_idx, idx)
    if silicon_atoms is None:
        return None
    return _prefix_candidate(
        "t-butyl(dimethyl)silanoxy",
        {idx, *silicon_atoms},
        idx,
        parent_idx,
        "tbdms-silanoxy",
    )


def _tbdms_atoms(mol, silicon_idx: int, blocked_idx: int) -> set[int] | None:
    silicon = mol.GetAtomWithIdx(silicon_idx)
    if silicon.GetSymbol() != "Si" or silicon.GetFormalCharge() != 0 or silicon.IsInRing() or silicon.GetIsotope():
        return None
    carbon_ligands = [
        nbr.GetIdx()
        for nbr in silicon.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetSymbol() == "C"
    ]
    if len(carbon_ligands) != 3:
        return None
    methyls = [carbon_idx for carbon_idx in carbon_ligands if _is_plain_terminal_methyl(mol, carbon_idx, silicon_idx)]
    tert_butyls = [
        carbon_idx
        for carbon_idx in carbon_ligands
        if not _is_plain_terminal_methyl(mol, carbon_idx, silicon_idx)
        and _tert_butyl_ligand_atoms(mol, carbon_idx, silicon_idx) is not None
    ]
    if len(methyls) != 2 or len(tert_butyls) != 1:
        return None
    tert_atoms = _tert_butyl_ligand_atoms(mol, tert_butyls[0], silicon_idx)
    if tert_atoms is None:
        return None
    return {silicon_idx, *methyls, *tert_atoms}


def _tert_butyl_ligand_atoms(mol, carbon_idx: int, center_idx: int) -> set[int] | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    if _atom_has_assigned_stereo(carbon) or _atom_has_isotopic_hydrogen(carbon):
        return None
    bond = mol.GetBondBetweenAtoms(carbon_idx, center_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    methyls = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != center_idx and nbr.GetSymbol() == "C"
    ]
    if len(methyls) != 3:
        return None
    if not all(_is_plain_terminal_methyl(mol, methyl_idx, carbon_idx) for methyl_idx in methyls):
        return None
    return {carbon_idx, *methyls}


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


def _oxo_methyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
        return None
    if _atom_has_isotopic_hydrogen(atom):
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
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or _formyl_prefix_allowed(mol, idx, parent_idx):
        return None
    isotope = atom.GetIsotope()
    text = f"1-oxo-methyl-{isotope}C" if isotope else "1-oxo-methyl"
    return _prefix_candidate(text, {idx, double_oxygens[0]}, idx, parent_idx, "oxo-methyl")


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
    retained = {"ethen-1-yl": "acryloyl", "methyl": "acetyl"}.get(ligand.text)
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


def _azabicyclo221_heptane_carbonyl_prefix(
    mol,
    idx: int,
    parent_idx: int,
    Chem=None,
) -> PrefixCandidate | None:
    carbon = mol.GetAtomWithIdx(idx)
    if carbon.GetSymbol() != "C" or carbon.IsInRing() or carbon.GetFormalCharge() != 0 or carbon.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygen_idx = None
    amide_nitrogen_idx = None
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond is None:
            return None
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if oxygen_idx is not None or nbr.GetFormalCharge() != 0 or nbr.GetDegree() != 1:
                return None
            oxygen_idx = nidx
            continue
        if nbr.GetSymbol() == "N" and bond.GetBondTypeAsDouble() == 1.0 and nbr.IsInRing():
            if amide_nitrogen_idx is not None or nbr.GetFormalCharge() != 0:
                return None
            amide_nitrogen_idx = nidx
            continue
        return None
    if oxygen_idx is None or amide_nitrogen_idx is None:
        return None

    core = _ring_component_from_atom(mol, amide_nitrogen_idx)
    if len(core) != 7:
        return None
    if amide_nitrogen_idx not in core:
        return None
    if _symbol_count(mol, core) != {"C": 6, "N": 1}:
        return None
    if any(mol.GetAtomWithIdx(atom_idx).GetFormalCharge() != 0 for atom_idx in core):
        return None

    core_degrees = {
        atom_idx: sum(1 for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetIdx() in core)
        for atom_idx in core
    }
    bridgeheads = [atom_idx for atom_idx, degree in core_degrees.items() if degree == 3]
    if len(bridgeheads) != 2:
        return None
    if core_degrees.get(amide_nitrogen_idx) != 2:
        return None
    amide_core_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(amide_nitrogen_idx).GetNeighbors()
        if nbr.GetIdx() in core
    ]
    locant_one_candidates = [atom_idx for atom_idx in amide_core_neighbors if atom_idx in bridgeheads]
    if len(locant_one_candidates) != 1:
        return None
    locant_one_idx = locant_one_candidates[0]
    locant_four_idx = next(atom_idx for atom_idx in bridgeheads if atom_idx != locant_one_idx)

    amino_records = []
    for atom_idx in core:
        atom = mol.GetAtomWithIdx(atom_idx)
        external = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() not in core and nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
        ]
        if atom_idx == amide_nitrogen_idx:
            if any(nbr.GetIdx() not in core and nbr.GetIdx() != idx for nbr in atom.GetNeighbors()):
                return None
            continue
        if not external:
            continue
        if len(external) != 1:
            return None
        amino_idx = external[0]
        if not _is_terminal_amino(mol, amino_idx, atom_idx):
            return None
        amino_records.append((atom_idx, amino_idx))
    if len(amino_records) != 1:
        return None
    locant_seven_idx, amino_idx = amino_records[0]
    if locant_seven_idx not in core or locant_seven_idx in {locant_one_idx, locant_four_idx}:
        return None
    if not _is_221_bicyclic_core(mol, core, locant_one_idx, locant_four_idx):
        return None

    stereo_prefix = _azabicyclo221_carbonyl_stereo_prefix(
        mol,
        Chem,
        ((1, locant_one_idx), (4, locant_four_idx), (7, locant_seven_idx)),
    )
    text = f"{stereo_prefix}7-amino-2-azabicyclo[2.2.1]heptane-2-carbonyl"
    return _prefix_candidate(
        text,
        {idx, oxygen_idx, amino_idx, *core},
        idx,
        parent_idx,
        "azabicyclo221-amino-carbonyl",
    )


def _ring_component_from_atom(mol, start_idx: int) -> set[int]:
    if not mol.GetAtomWithIdx(start_idx).IsInRing():
        return set()
    seen = {start_idx}
    stack = [start_idx]
    while stack:
        atom_idx = stack.pop()
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in seen or not nbr.IsInRing():
                continue
            seen.add(nidx)
            stack.append(nidx)
    return seen


def _symbol_count(mol, atoms: set[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom_idx in atoms:
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _is_terminal_amino(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


def _is_221_bicyclic_core(
    mol,
    core: set[int],
    locant_one_idx: int,
    locant_four_idx: int,
) -> bool:
    paths = _simple_paths_within_atoms(mol, core, locant_one_idx, locant_four_idx, max_paths=4)
    if len(paths) != 3:
        return False
    bridge_lengths = sorted(len(path) - 2 for path in paths)
    return bridge_lengths == [1, 2, 2]


def _simple_paths_within_atoms(
    mol,
    atoms: set[int],
    start_idx: int,
    end_idx: int,
    *,
    max_paths: int,
) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    stack = [(start_idx, (start_idx,))]
    while stack:
        current, path = stack.pop()
        if len(paths) >= max_paths:
            break
        if current == end_idx:
            paths.append(path)
            continue
        for nbr in mol.GetAtomWithIdx(current).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in atoms or nidx in path:
                continue
            stack.append((nidx, (*path, nidx)))
    return paths


def _azabicyclo221_carbonyl_stereo_prefix(
    mol,
    Chem,
    locanted_atoms: tuple[tuple[int, int], ...],
) -> str:
    if Chem is None:
        return ""
    descriptors_by_atom = dict(assigned_atom_stereochemistry(mol, Chem, atom_order="idx"))
    descriptors = [
        f"{locant}{descriptors_by_atom[atom_idx]}"
        for locant, atom_idx in locanted_atoms
        if atom_idx in descriptors_by_atom
    ]
    return f"({','.join(descriptors)})-" if descriptors else ""


def _retained_amido_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    acyl = _retained_acyl_prefix(mol, heavy[0], idx)
    if acyl is None or acyl.text != "acetyl":
        return None
    return _prefix_candidate(
        "acetamido",
        {idx, *acyl.atoms},
        idx,
        parent_idx,
        "acetamido",
    )


def _alkoxycarbonyl_amino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    ligand = _retained_alkoxycarbonyl_prefix(mol, heavy[0], idx)
    if ligand is None or ligand.text not in {"methoxycarbonyl", "ethoxycarbonyl"}:
        return None
    text = ligand.text + "amino"
    return _prefix_candidate(text, {idx, *ligand.atoms}, idx, parent_idx, text)


def _trifluoroacetamido_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    acyl = _trifluoroacetyl_prefix(mol, heavy[0], idx)
    if acyl is None:
        return None
    return _prefix_candidate("trifluoroacetamido", {idx, *acyl.atoms}, idx, parent_idx, "trifluoroacetamido")


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
    nitrogen_ligands = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
    ]
    if len(nitrogen_ligands) > 2:
        return None
    benzyl_ligand = _substituted_benzylaminocarbonyl_ligand(mol, nitrogen_ligands, nitrogens[0])
    if benzyl_ligand is not None:
        return _prefix_candidate(
            f"(({benzyl_ligand.text})amino)carbonyl",
            {idx, double_oxygens[0], nitrogens[0], *benzyl_ligand.atoms},
            idx,
            parent_idx,
            "substituted-benzylaminocarbonyl",
        )
    if not all(_is_plain_terminal_methyl(mol, ligand_idx, nitrogens[0]) for ligand_idx in nitrogen_ligands):
        return None
    names = {0: "carbamoyl", 1: "methylcarbamoyl", 2: "dimethylcarbamoyl"}
    name = names[len(nitrogen_ligands)]
    return _prefix_candidate(
        name,
        {idx, double_oxygens[0], nitrogens[0], *nitrogen_ligands},
        idx,
        parent_idx,
        name,
    )


def _substituted_benzylaminocarbonyl_ligand(
    mol,
    nitrogen_ligands: list[int],
    nitrogen_idx: int,
) -> PrefixCandidate | None:
    if len(nitrogen_ligands) != 1:
        return None
    candidates = simple_external_prefix_candidates(mol, nitrogen_ligands[0], nitrogen_idx)
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    if not _safe_substituted_benzylaminocarbonyl_ligand(candidate.text):
        return None
    return candidate


def _safe_substituted_benzylaminocarbonyl_ligand(text: str) -> bool:
    if not text.endswith("benzyl") or not text[:1].isdigit():
        return False
    if len(text) > 40:
        return False
    if not all(ch.isalnum() or ch in ",-" for ch in text):
        return False
    return any(term in text for term in ("fluoro", "chloro", "bromo", "methyl", "methoxy"))


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
        "methyl": "methoxycarbonyl",
        "ethyl": "ethoxycarbonyl",
        "benzyl": "cbz",
        "t-butyl": "tert-butoxycarbonyl",
    }
    name = names.get(text)
    if name is None:
        return None
    return _prefix_candidate(name, {idx, double_oxygens[0], single_oxygens[0], *alkyl_atoms}, idx, parent_idx, name)


def _trifluoroacetyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
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
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope() or nbr.IsInRing():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondTypeAsDouble() == 1.0:
            carbon_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(carbon_ligands) != 1:
        return None
    trifluoromethyl_atoms = _terminal_trifluoromethyl_atoms(mol, carbon_ligands[0], idx)
    if trifluoromethyl_atoms is None:
        return None
    return _prefix_candidate(
        "trifluoroacetyl",
        {idx, double_oxygens[0], *trifluoromethyl_atoms},
        idx,
        parent_idx,
        "trifluoroacetyl",
    )


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


def _terminal_trifluoromethyl_atoms(mol, carbon_idx: int, parent_idx: int) -> set[int] | None:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return None
    if _atom_has_assigned_stereo(carbon) or _atom_has_isotopic_hydrogen(carbon):
        return None
    parent_bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    fluorines = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "F"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and not nbr.GetIsotope()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 1.0
            and not any(extra.GetIdx() != carbon_idx and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors())
        ):
            fluorines.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(fluorines) != 3:
        return None
    return {carbon_idx, *fluorines}


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


def _borono_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "B" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        if _is_terminal_hydroxy_oxygen(mol, nidx, idx):
            oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(oxygens) != 2:
        return None
    return _prefix_candidate("borono", {idx, *oxygens}, idx, parent_idx, "borono")


def _dihaloboryl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "B" or atom.GetFormalCharge() != 0 or atom.GetIsotope() or atom.IsInRing():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    halogens = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() in HALO
            and nbr.GetFormalCharge() == 0
            and not nbr.GetIsotope()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 1.0
            and not any(other.GetIdx() != idx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
        ):
            halogens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(halogens) != 2:
        return None
    symbols = [mol.GetAtomWithIdx(nidx).GetSymbol() for nidx in halogens]
    if len(set(symbols)) != 1:
        return None
    text = MULT[2] + HALO[symbols[0]] + "boryl"
    return _prefix_candidate(text, {idx, *halogens}, idx, parent_idx, "dihaloboryl")


def _pinacol_boronate_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "B" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxygen_indices = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondTypeAsDouble() == 1.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            oxygen_indices.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(oxygen_indices) != 2:
        return None
    ring_carbons = []
    for oxygen_idx in oxygen_indices:
        carbon_idx = _single_heavy_neighbor_with_symbol(mol, oxygen_idx, idx, "C")
        if carbon_idx is None:
            return None
        ring_carbons.append(carbon_idx)
    if len(set(ring_carbons)) != 2:
        return None
    carbon_bond = mol.GetBondBetweenAtoms(ring_carbons[0], ring_carbons[1])
    if carbon_bond is None or carbon_bond.GetBondTypeAsDouble() != 1.0:
        return None
    methyl_atoms = []
    for carbon_idx in ring_carbons:
        carbon = mol.GetAtomWithIdx(carbon_idx)
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.GetIsotope():
            return None
        methyls = []
        for nbr in carbon.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring_carbons or nidx in oxygen_indices:
                continue
            if not _is_plain_terminal_methyl(mol, nidx, carbon_idx):
                return None
            methyls.append(nidx)
        if len(methyls) != 2:
            return None
        methyl_atoms.extend(methyls)
    return _prefix_candidate(
        "4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl",
        {idx, *oxygen_indices, *ring_carbons, *methyl_atoms},
        idx,
        parent_idx,
        "pinacol-boronate",
    )


def _bis_oxidanidyl_phosphanyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0 or atom.GetIsotope() or atom.IsInRing():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    oxidanidyl = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() < 0
            and not nbr.GetIsotope()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 1.0
            and not any(other.GetIdx() != idx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
        ):
            oxidanidyl.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(oxidanidyl) != 2:
        return None
    return _prefix_candidate(
        "bis(oxidanidyl)phosphanyl",
        {idx, *oxidanidyl},
        idx,
        parent_idx,
        "bis-oxidanidyl-phosphanyl",
    )


def _structured_phosphoryl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0 or atom.GetIsotope() or atom.IsInRing():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygen = []
    ligand_records = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and not nbr.GetIsotope()
            and not nbr.IsInRing()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
            and not any(other.GetIdx() != idx and other.GetAtomicNum() > 1 for other in nbr.GetNeighbors())
        ):
            double_oxygen.append(nidx)
            continue
        ligand = _phosphoryl_ligand_candidate(mol, nidx, idx)
        if ligand is None:
            return None
        ligand_records.append(ligand)
    if len(double_oxygen) != 1 or len(ligand_records) != 2:
        return None
    first, second = ligand_records
    first_type = mol.GetAtomWithIdx(first.attachment_atom).GetAtomicNum()
    second_type = mol.GetAtomWithIdx(second.attachment_atom).GetAtomicNum()
    if {first_type, second_type} == {6, 8}:
        carbon, oxygen = (first, second) if first_type == 6 else (second, first)
        ligand_text = f"{_phosphoryl_ligand_unit(carbon.text)}({oxygen.text})"
    else:
        ligand_text = _phosphoryl_ligand_text([ligand.text for ligand in ligand_records])
    if not ligand_text:
        return None
    atoms = {idx, double_oxygen[0]}
    for ligand in ligand_records:
        atoms.update(ligand.atoms)
    return _prefix_candidate(
        ligand_text + "phosphoryl",
        atoms,
        idx,
        parent_idx,
        "structured-phosphoryl",
    )


def _phosphoryl_ligand_candidate(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    candidates = simple_external_prefix_candidates(mol, idx, parent_idx)
    if candidates:
        candidate = candidates[0]
    elif atom.IsInRing():
        candidate = _simple_ring_prefix(mol, idx, parent_idx)
    else:
        candidate = None
    if candidate is None:
        return None
    if not _safe_phosphoryl_ligand_text(candidate.text):
        return None
    return candidate


def _safe_phosphoryl_ligand_text(text: str) -> bool:
    if not text or len(text) > 70:
        return False
    if " " in text or "[" in text or "]" in text:
        return False
    if not all(ch.isalnum() or ch in ",().'-" for ch in text):
        return False
    forbidden = ("carbonyl", "carbam", "phosph", "sulfon", "sulfin")
    return not any(fragment in text for fragment in forbidden)


def _phosphoryl_ligand_text(ligands: list[str]) -> str:
    counts = Counter(ligands)
    pieces = []
    for ligand in sorted(counts, key=_phosphoryl_ligand_sort_key):
        count = counts[ligand]
        unit = _phosphoryl_ligand_unit(ligand)
        if count == 1:
            pieces.append(unit)
        elif _needs_complex_phosphoryl_ligand(ligand):
            pieces.append(f"{MULT.get(count, '')}({unit})")
        else:
            pieces.append(MULT.get(count, "") + ligand)
    return "".join(pieces)


def _phosphoryl_ligand_unit(ligand: str) -> str:
    return f"({ligand})" if _needs_complex_phosphoryl_ligand(ligand) else ligand


def _needs_complex_phosphoryl_ligand(ligand: str) -> bool:
    return "-" in ligand or ligand[:1].isdigit() or ligand.startswith("(")


def _phosphoryl_ligand_sort_key(ligand: str) -> tuple[int, str]:
    simple_alkyl = {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
    }
    if ligand in simple_alkyl:
        return (0, ligand)
    if _needs_complex_phosphoryl_ligand(ligand):
        return (2, ligand)
    return (1, ligand)


def _single_heavy_neighbor_with_symbol(
    mol,
    atom_idx: int,
    blocked_idx: int,
    symbol: str,
) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
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
    if neighbor.GetSymbol() != symbol or bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return None
    return neighbor_idx


def _is_terminal_hydroxy_oxygen(mol, oxygen_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(oxygen_idx)
    if atom.GetSymbol() != "O" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(oxygen_idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


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
    if ligand is None or ligand.text != "tert-butoxycarbonyl":
        return None
    if not _bocamino_prefix_allowed(mol, idx, parent_idx):
        return None
    return _prefix_candidate("(tert-butoxycarbonyl)amino", {idx, *ligand.atoms}, idx, parent_idx, "bocamino")


def _long_bocamino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    ligand = _retained_alkoxycarbonyl_prefix(mol, heavy[0], idx)
    if ligand is None or ligand.text != "tert-butoxycarbonyl":
        return None
    parent = mol.GetAtomWithIdx(parent_idx)
    if not (parent.GetSymbol() == "C" and parent.IsInRing() and not parent.GetIsAromatic()):
        return None
    return _prefix_candidate(
        "(tert-butoxycarbonyl)amino",
        {idx, *ligand.atoms},
        idx,
        parent_idx,
        "long-bocamino",
    )


def _mesylamino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    ligand = _mesyl_prefix(mol, heavy[0], idx)
    if ligand is None:
        return None
    return _prefix_candidate(f"({ligand.text})amino", {idx, *ligand.atoms}, idx, parent_idx, "mesylamino")


def _sulfonylamino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() != "C":
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    sulfur_idx = heavy[0]
    ligand = _sulfonyl_ligand_prefix(mol, sulfur_idx, idx)
    if ligand is None:
        return None
    if not _safe_sulfonylamino_ligand_text(ligand.text):
        return None
    return _prefix_candidate(
        ligand.text + "sulfonylamino",
        {idx, *ligand.atoms},
        idx,
        parent_idx,
        "sulfonylamino",
    )


def _sulfonyl_ligand_prefix(mol, sulfur_idx: int, nitrogen_idx: int) -> PrefixCandidate | None:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0 or sulfur.IsInRing() or sulfur.GetIsotope():
        return None
    ns_bond = mol.GetBondBetweenAtoms(sulfur_idx, nitrogen_idx)
    if ns_bond is None or ns_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    ligand_indices = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and not nbr.GetIsotope()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
            and not any(extra.GetIdx() != sulfur_idx and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors())
        ):
            double_oxygens.append(nidx)
        elif bond is not None and bond.GetBondTypeAsDouble() == 1.0 and nbr.GetAtomicNum() > 1:
            ligand_indices.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 2 or len(ligand_indices) != 1:
        return None
    ligand_idx = ligand_indices[0]
    candidates = simple_external_prefix_candidates(mol, ligand_idx, sulfur_idx)
    if candidates:
        ligand = candidates[0]
    elif mol.GetAtomWithIdx(ligand_idx).IsInRing():
        ligand = _simple_ring_prefix(mol, ligand_idx, sulfur_idx)
    else:
        ligand = None
    if ligand is None:
        return None
    return _prefix_candidate(
        ligand.text,
        {sulfur_idx, *double_oxygens, *ligand.atoms},
        sulfur_idx,
        nitrogen_idx,
        "sulfonyl-ligand",
    )


def _safe_sulfonylamino_ligand_text(text: str) -> bool:
    if not text or len(text) > 90:
        return False
    if " " in text or "[" in text or "]" in text or "(" in text or ")" in text:
        return False
    if not text.endswith(("yl", "yl'", "furyl", "thienyl")):
        return False
    if not all(ch.isalnum() or ch in ",.'-" for ch in text):
        return False
    forbidden = ("carbonyl", "carbam", "phosph", "sulfon", "sulfin", "sulf")
    return not any(fragment in text for fragment in forbidden)


def _ethenylcarbonyl_amino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    if _atom_has_assigned_stereo(atom) or _atom_has_isotopic_hydrogen(atom):
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]
    if len(heavy) != 1:
        return None
    acyl = _retained_acyl_prefix(mol, heavy[0], idx)
    if acyl is None or acyl.text != "acryloyl":
        return None
    return _prefix_candidate(
        "(ethen-1-ylcarbonyl)amino",
        {idx, *acyl.atoms},
        idx,
        parent_idx,
        "ethenylcarbonyl-amino",
    )


def _sulfinylamino_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    sulfur_idx = None
    oxygen_idx = None
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            sulfur_idx is None
            and nbr.GetSymbol() == "S"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
        ):
            sulfur_idx = nidx
        else:
            return None
    if sulfur_idx is None:
        return None
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx or nbr.GetAtomicNum() <= 1:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if (
            oxygen_idx is None
            and nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and bond is not None
            and bond.GetBondTypeAsDouble() == 2.0
        ):
            oxygen_idx = nidx
        else:
            return None
    if oxygen_idx is None:
        return None
    return _prefix_candidate("sulfinylamino", {idx, sulfur_idx, oxygen_idx}, idx, parent_idx, "sulfinylamino")


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
    return _prefix_candidate("methanesulfonyl", {idx, *double_oxygens, methyl_ligands[0]}, idx, parent_idx, "mesyl")


def _aminosulfonyl_prefix(mol, idx: int, parent_idx: int) -> PrefixCandidate | None:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return None
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondTypeAsDouble() != 1.0:
        return None
    double_oxygens = []
    nitrogen_ligands = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondTypeAsDouble() == 2.0:
            if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
                return None
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "N" and bond.GetBondTypeAsDouble() == 1.0:
            nitrogen_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 2 or len(nitrogen_ligands) != 1:
        return None
    nitrogen = simple_external_prefix_candidates(mol, nitrogen_ligands[0], idx)
    if len(nitrogen) != 1 or nitrogen[0].text != "amino":
        return None
    return _prefix_candidate(
        "aminosulfonyl",
        {idx, *double_oxygens, *nitrogen[0].atoms},
        idx,
        parent_idx,
        "aminosulfonyl",
    )


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
    if len(ring) == 5 and all(atom.GetIsAromatic() for atom in atoms):
        heteroaryl = _simple_five_member_heteroaryl_prefix(mol, ring, idx, parent_idx)
        if heteroaryl is not None:
            return heteroaryl
    if any(atom.GetSymbol() != "C" for atom in atoms):
        unsaturated_oxa = _unsaturated_oxa_cyclohexenyl_prefix(mol, ring_set, idx, parent_idx)
        if unsaturated_oxa is not None:
            return unsaturated_oxa
        return _simple_saturated_heterocycle_prefix(mol, ring, idx, parent_idx)
    if any(atom.GetIsAromatic() for atom in atoms):
        return None
    if not _ring_bonds_are_single(mol, ring):
        return None
    difluoro_cycloalkyl = _gem_difluoro_cycloalkyl_prefix(mol, ring_set, idx, parent_idx)
    if difluoro_cycloalkyl is not None:
        return difluoro_cycloalkyl
    if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
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
    counts = _ring_symbol_counts(mol, ring)
    if len(ring) == 6 and counts == {"C": 5, "N": 1}:
        dioxopiperidyl = _dioxopiperidyl_prefix(mol, ring_set, idx, parent_idx)
        if dioxopiperidyl is not None:
            text, atoms = dioxopiperidyl
            return _prefix_candidate(text, atoms, idx, parent_idx, "dioxopiperidyl-ring")
    if len(ring) == 6 and counts == {"C": 4, "N": 2}:
        boc_piperazino = _boc_piperazino_prefix(mol, ring_set, idx, parent_idx)
        if boc_piperazino is not None:
            text, atoms = boc_piperazino
            return _prefix_candidate(text, atoms, idx, parent_idx, "boc-piperazino-ring")
        n_methyl_piperazino = _n_methyl_piperazino_prefix(mol, ring_set, idx, parent_idx)
        if n_methyl_piperazino is not None:
            text, atoms = n_methyl_piperazino
            return _prefix_candidate(text, atoms, idx, parent_idx, "n-methyl-piperazino-ring")
    if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
        return None
    text = None
    if len(ring) == 6 and counts == {"C": 4, "N": 1, "O": 1}:
        text = _morpholine_prefix(mol, ring_set, idx)
    elif len(ring) == 6 and counts == {"C": 4, "N": 2}:
        text = _piperazine_prefix(mol, ring_set, idx)
    elif len(ring) == 6 and counts == {"C": 5, "N": 1}:
        text = _piperidine_prefix(mol, ring_set, idx)
    elif len(ring) == 5 and counts == {"C": 4, "N": 1}:
        text = "pyrrolidin-1-yl" if mol.GetAtomWithIdx(idx).GetSymbol() == "N" else None
    elif len(ring) in {4, 5, 6} and counts == {"C": len(ring) - 1, "O": 1}:
        text = _simple_oxa_ring_prefix(mol, ring_set, idx, len(ring))
    if text is None:
        return None
    return _prefix_candidate(text, ring_set, idx, parent_idx, "simple-saturated-heterocycle")


def _unsaturated_oxa_cyclohexenyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    if len(ring_set) != 6:
        return None
    oxygens = _ring_atoms_by_symbol(mol, ring_set, "O")
    if len(oxygens) != 1 or idx == oxygens[0]:
        return None
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() not in {"C", "O"} for atom_idx in ring_set):
        return None
    if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
        return None
    double_bonds = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in ring_set or end not in ring_set:
            continue
        order = bond.GetBondTypeAsDouble()
        if order == 2.0:
            double_bonds.append(frozenset((begin, end)))
        elif order != 1.0:
            return None
    if len(double_bonds) != 1:
        return None
    oxygen_idx = oxygens[0]
    ring_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()
        if nbr.GetIdx() in ring_set
    ]
    if len(ring_neighbors) != 2:
        return None
    for first_idx in ring_neighbors:
        seq = _oriented_ring_sequence(mol, ring_set, oxygen_idx, first_idx)
        if len(seq) != 6 or set(seq) != ring_set:
            continue
        if seq[3] != idx:
            continue
        if frozenset((seq[2], seq[3])) != double_bonds[0]:
            continue
        return _prefix_candidate(
            "3,6-dihydro-2H-pyran-4-yl",
            ring_set,
            idx,
            parent_idx,
            "unsaturated-oxa-cyclohexenyl-ring",
        )
    return None


def _dioxopiperidyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> tuple[str, set[int]] | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 1:
        return None
    nitrogen = nitrogens[0]
    if mol.GetAtomWithIdx(nitrogen).GetTotalNumHs() != 1:
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    carbonyl_carbons = []
    carbonyl_oxygens = []
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
            continue
        if atom_idx == nitrogen:
            if neighbors:
                return None
            continue
        if not neighbors:
            continue
        if len(neighbors) != 1:
            return None
        oxygen_idx = neighbors[0]
        oxygen = mol.GetAtomWithIdx(oxygen_idx)
        bond = mol.GetBondBetweenAtoms(atom_idx, oxygen_idx)
        if (
            oxygen.GetSymbol() != "O"
            or oxygen.GetFormalCharge() != 0
            or oxygen.GetIsotope()
            or oxygen.IsInRing()
            or bond is None
            or bond.GetBondTypeAsDouble() != 2.0
        ):
            return None
        carbonyl_carbons.append(atom_idx)
        carbonyl_oxygens.append(oxygen_idx)
    if len(carbonyl_carbons) != 2:
        return None
    if not all(_ring_atoms_adjacent(mol, ring_set, nitrogen, carbonyl_idx) for carbonyl_idx in carbonyl_carbons):
        return None
    c1, c2 = carbonyl_carbons
    path = _ring_path_between(mol, ring_set, c1, c2, forbidden_edge=(c1, nitrogen))
    if len(path) != 5 or idx not in path[1:-1]:
        return None
    locant = path.index(idx) + 2
    locant = min(locant, 8 - locant)
    if locant not in {3, 4}:
        return None
    return f"2,6-dioxopiperidin-{locant}-yl", ring_set | set(carbonyl_oxygens)


def _n_methyl_piperazino_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> tuple[str, set[int]] | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "N":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2:
        return None
    other_nitrogen = nitrogens[0] if idx == nitrogens[1] else nitrogens[1]
    if _ring_distance(mol, ring_set, idx, other_nitrogen) != 3:
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
        elif atom_idx == other_nitrogen:
            if len(neighbors) != 1:
                return None
        elif neighbors:
            return None
    methyl_idx = external[other_nitrogen][0]
    if not _is_plain_terminal_methyl(mol, methyl_idx, other_nitrogen):
        return None
    return "4-methylpiperazin-1-yl", ring_set | {methyl_idx}


def _boc_piperazino_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> tuple[str, set[int]] | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "N":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2:
        return None
    other_nitrogen = nitrogens[0] if idx == nitrogens[1] else nitrogens[1]
    if _ring_distance(mol, ring_set, idx, other_nitrogen) != 3:
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
        elif atom_idx == other_nitrogen:
            if len(neighbors) != 1:
                return None
        elif neighbors:
            return None
    boc = _retained_alkoxycarbonyl_prefix(mol, external[other_nitrogen][0], other_nitrogen)
    if boc is None or boc.text != "tert-butoxycarbonyl":
        return None
    return "4-(tert-butoxycarbonyl)piperazin-1-yl", ring_set | set(boc.atoms)


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
    return "morpholin-4-yl" if distance == 3 else None


def _piperazine_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "N":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2:
        return None
    other = nitrogens[0] if idx == nitrogens[1] else nitrogens[1]
    distance = _ring_distance(mol, ring_set, idx, other)
    return "piperazin-1-yl" if distance == 3 else None


def _piperidine_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 1:
        return None
    if idx == nitrogens[0]:
        return "piperidin-1-yl"
    distance = _ring_distance(mol, ring_set, nitrogens[0], idx)
    if distance is None:
        return None
    locant = distance + 1
    return f"piperidin-{locant}-yl" if locant in {2, 3, 4} else None


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


def _simple_five_member_heteroaryl_prefix(
    mol,
    ring: tuple[int, ...],
    idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    ring_set = set(ring)
    counts = _ring_symbol_counts(mol, ring)
    if counts == {"C": 3, "N": 2}:
        n_methyl_pyrazolyl = _n_methyl_pyrazolyl_prefix(mol, ring_set, idx, parent_idx)
        if n_methyl_pyrazolyl is not None:
            text, atoms = n_methyl_pyrazolyl
            return _prefix_candidate(text, atoms, idx, parent_idx, "n-methyl-pyrazolyl-ring")
    text = None
    if counts == {"C": 4, "O": 1}:
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _single_heteroaryl_prefix(mol, ring_set, idx, "O", "furyl")
    elif counts == {"C": 4, "S": 1}:
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _single_heteroaryl_prefix(mol, ring_set, idx, "S", "thienyl")
    elif counts == {"C": 4, "N": 1}:
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _pyrrole_prefix(mol, ring_set, idx)
    elif counts == {"C": 3, "N": 2}:
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _diazole_prefix(mol, ring_set, idx)
    elif counts == {"C": 3, "N": 1, "O": 1}:
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _azole_prefix(mol, ring_set, idx, "O", "oxazol")
    elif counts == {"C": 3, "N": 1, "S": 1}:
        amino_thiazolyl = _amino_thiazolyl_prefix(mol, ring_set, idx, parent_idx)
        if amino_thiazolyl is not None:
            text, atoms = amino_thiazolyl
            return _prefix_candidate(text, atoms, idx, parent_idx, "amino-thiazolyl-ring")
        if not _ring_has_only_parent_external(mol, ring_set, idx, parent_idx):
            return None
        text = _azole_prefix(mol, ring_set, idx, "S", "thiazol")
    if text is None:
        return None
    return _prefix_candidate(text, ring_set, idx, parent_idx, "simple-five-member-heteroaryl-ring")


def _n_methyl_pyrazolyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> tuple[str, set[int]] | None:
    if mol.GetAtomWithIdx(idx).GetSymbol() == "N":
        return None
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2:
        return None
    n1, n2 = nitrogens
    if not _ring_atoms_adjacent(mol, ring_set, n1, n2):
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
        elif atom_idx not in nitrogens and neighbors:
            return None
    substituted_nitrogens = [nidx for nidx in nitrogens if external[nidx]]
    if len(substituted_nitrogens) != 1:
        return None
    substituted_nitrogen = substituted_nitrogens[0]
    methyl_idx = external[substituted_nitrogen][0]
    if not _is_plain_terminal_methyl(mol, methyl_idx, substituted_nitrogen):
        return None
    plain_nitrogen = n1 if substituted_nitrogen == n2 else n2
    if external[plain_nitrogen]:
        return None
    if any(mol.GetAtomWithIdx(nidx).GetTotalNumHs() for nidx in nitrogens):
        return None
    locants = _pyrazole_locants(mol, ring_set, substituted_nitrogen, plain_nitrogen)
    locant = locants.get(idx)
    if locant not in {3, 4, 5}:
        return None
    return f"1-methylpyrazol-{locant}-yl", ring_set | {methyl_idx}


def _ring_external_heavy_neighbors_by_atom(mol, ring_set: set[int]) -> dict[int, list[int]]:
    external: dict[int, list[int]] = {}
    for atom_idx in ring_set:
        external[atom_idx] = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1
        ]
    return external


def _amino_thiazolyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> tuple[str, set[int]] | None:
    sulfurs = _ring_atoms_by_symbol(mol, ring_set, "S")
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(sulfurs) != 1 or len(nitrogens) != 1 or idx in {sulfurs[0], nitrogens[0]}:
        return None
    locants = _one_three_azole_locants(mol, ring_set, sulfurs[0], nitrogens[0])
    if locants.get(idx) != 4:
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    amino_atoms: set[int] | None = None
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
            continue
        if locants.get(atom_idx) == 2:
            if len(neighbors) != 1:
                return None
            candidates = simple_external_prefix_candidates(mol, neighbors[0], atom_idx)
            if len(candidates) != 1 or candidates[0].text != "amino":
                return None
            amino_atoms = set(candidates[0].atoms)
            continue
        if neighbors:
            return None
    if amino_atoms is None:
        return None
    return "2-aminothiazol-4-yl", ring_set | amino_atoms


def _single_heteroaryl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    hetero_symbol: str,
    suffix: str,
) -> str | None:
    hetero_atoms = _ring_atoms_by_symbol(mol, ring_set, hetero_symbol)
    if len(hetero_atoms) != 1 or idx == hetero_atoms[0]:
        return None
    locant = _min_ring_distance_locant(mol, ring_set, hetero_atoms[0], idx)
    return f"{locant}-{suffix}" if locant in {2, 3} else None


def _pyrrole_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 1 or idx == nitrogens[0]:
        return None
    nitrogen = mol.GetAtomWithIdx(nitrogens[0])
    if nitrogen.GetTotalNumHs() != 1:
        return None
    locant = _min_ring_distance_locant(mol, ring_set, nitrogens[0], idx)
    return f"1H-pyrrol-{locant}-yl" if locant in {2, 3} else None


def _diazole_prefix(mol, ring_set: set[int], idx: int) -> str | None:
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(nitrogens) != 2 or idx in nitrogens:
        return None
    n_h = [nidx for nidx in nitrogens if mol.GetAtomWithIdx(nidx).GetTotalNumHs() == 1]
    n_plain = [nidx for nidx in nitrogens if mol.GetAtomWithIdx(nidx).GetTotalNumHs() == 0]
    if len(n_h) != 1 or len(n_plain) != 1:
        return None
    nh_idx = n_h[0]
    plain_idx = n_plain[0]
    if _ring_atoms_adjacent(mol, ring_set, nh_idx, plain_idx):
        locants = _pyrazole_locants(mol, ring_set, nh_idx, plain_idx)
        parent = "pyrazol"
    else:
        locants = _imidazole_locants(mol, ring_set, nh_idx, plain_idx)
        parent = "imidazol"
    locant = locants.get(idx)
    return f"1H-{parent}-{locant}-yl" if locant in {2, 3, 4, 5} else None


def _azole_prefix(mol, ring_set: set[int], idx: int, hetero_symbol: str, parent: str) -> str | None:
    hetero_atoms = _ring_atoms_by_symbol(mol, ring_set, hetero_symbol)
    nitrogens = _ring_atoms_by_symbol(mol, ring_set, "N")
    if len(hetero_atoms) != 1 or len(nitrogens) != 1 or idx in {hetero_atoms[0], nitrogens[0]}:
        return None
    locants = _one_three_azole_locants(mol, ring_set, hetero_atoms[0], nitrogens[0])
    locant = locants.get(idx)
    return f"{parent}-{locant}-yl" if locant in {2, 4, 5} else None


def _pyrazole_locants(mol, ring_set: set[int], nh_idx: int, plain_idx: int) -> dict[int, int]:
    locants = {nh_idx: 1, plain_idx: 2}
    path = _ring_path_between(mol, ring_set, plain_idx, nh_idx, forbidden_edge=(plain_idx, nh_idx))
    if len(path) == 5:
        for locant, atom_idx in enumerate(path[1:-1], start=3):
            locants[atom_idx] = locant
    return locants


def _imidazole_locants(mol, ring_set: set[int], nh_idx: int, plain_idx: int) -> dict[int, int]:
    locants = {nh_idx: 1, plain_idx: 3}
    short_path = _shortest_ring_path(mol, ring_set, nh_idx, plain_idx)
    if len(short_path) == 3:
        locants[short_path[1]] = 2
    long_path = _ring_path_between(
        mol,
        ring_set,
        plain_idx,
        nh_idx,
        forbidden_edge=(plain_idx, short_path[1]) if len(short_path) == 3 else None,
    )
    if len(long_path) == 4:
        locants[long_path[1]] = 4
        locants[long_path[2]] = 5
    return locants


def _one_three_azole_locants(mol, ring_set: set[int], hetero_idx: int, nitrogen_idx: int) -> dict[int, int]:
    locants = {hetero_idx: 1, nitrogen_idx: 3}
    short_path = _shortest_ring_path(mol, ring_set, hetero_idx, nitrogen_idx)
    if len(short_path) == 3:
        locants[short_path[1]] = 2
    long_path = _ring_path_between(
        mol,
        ring_set,
        nitrogen_idx,
        hetero_idx,
        forbidden_edge=(nitrogen_idx, short_path[1]) if len(short_path) == 3 else None,
    )
    if len(long_path) == 4:
        locants[long_path[1]] = 4
        locants[long_path[2]] = 5
    return locants


def _min_ring_distance_locant(mol, ring_set: set[int], start_idx: int, target_idx: int) -> int | None:
    distance = _ring_distance(mol, ring_set, start_idx, target_idx)
    return None if distance is None else distance + 1


def _ring_atoms_adjacent(mol, ring_set: set[int], left: int, right: int) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and left in ring_set and right in ring_set


def _shortest_ring_path(mol, ring_set: set[int], start_idx: int, target_idx: int) -> tuple[int, ...]:
    path = _ring_path_between(mol, ring_set, start_idx, target_idx)
    return path


def _ring_path_between(
    mol,
    ring_set: set[int],
    start_idx: int,
    target_idx: int,
    *,
    forbidden_edge: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    forbidden = frozenset(forbidden_edge or ())
    queue: list[tuple[int, tuple[int, ...]]] = [(start_idx, (start_idx,))]
    seen = {start_idx}
    while queue:
        atom_idx, path = queue.pop(0)
        if atom_idx == target_idx:
            return path
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in ring_set or nidx in seen:
                continue
            if forbidden and frozenset((atom_idx, nidx)) == forbidden:
                continue
            seen.add(nidx)
            queue.append((nidx, path + (nidx,)))
    return ()


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
    return _prefix_candidate(f"pyridin-{locant}-yl", ring_set, idx, parent_idx, "simple-pyridyl-ring")


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
    if len(substituents) >= 2:
        multi = _multisubstituted_phenyl_prefix(mol, ring_set, idx, parent_idx, substituents)
        if multi is not None:
            return multi
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


def _multisubstituted_phenyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
    substituents: list[tuple[int, PrefixCandidate]],
) -> PrefixCandidate | None:
    if not (2 <= len(substituents) <= 5):
        return None
    if any(candidate.text not in _PHENYL_SIMPLE_SUBSTITUENTS for _, candidate in substituents):
        return None
    locanted = _phenyl_multisubstituent_locants(mol, ring_set, idx, substituents)
    if locanted is None:
        return None
    prefixes = [(str(locant), candidate.text) for locant, _atom_idx, candidate in locanted]
    prefix_text = _format_prefixes(prefixes)
    if not prefix_text:
        return None
    atoms = set(ring_set)
    for _locant, _atom_idx, candidate in locanted:
        atoms.update(candidate.atoms)
    return _prefix_candidate(
        prefix_text + "phenyl",
        atoms,
        idx,
        parent_idx,
        "multi-substituted-aryl-ring",
    )


def _difluorophenyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
    substituents: list[tuple[int, PrefixCandidate]],
) -> PrefixCandidate | None:
    if len(substituents) != 2 or any(candidate.text != "fluoro" for _, candidate in substituents):
        return None
    locants = _phenyl_disubstituent_locants(mol, ring_set, idx, [atom_idx for atom_idx, _ in substituents])
    if locants is None or any(locant <= 1 for locant in locants):
        return None
    atoms = set(ring_set)
    for _, candidate in substituents:
        atoms.update(candidate.atoms)
    return _prefix_candidate(
        f"{','.join(str(locant) for locant in locants)}-difluorophenyl",
        atoms,
        idx,
        parent_idx,
        "difluoro-aryl-ring",
    )


def _phenyl_multisubstituent_locants(
    mol,
    ring_set: set[int],
    attachment_idx: int,
    substituents: list[tuple[int, PrefixCandidate]],
) -> tuple[tuple[int, int, PrefixCandidate], ...] | None:
    ring_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(attachment_idx).GetNeighbors()
        if nbr.GetIdx() in ring_set
    ]
    if len(ring_neighbors) != 2:
        return None
    options = []
    for first_idx in ring_neighbors:
        seq = _oriented_ring_sequence(mol, ring_set, attachment_idx, first_idx)
        if len(seq) != len(ring_set) or set(seq) != ring_set:
            continue
        entries = []
        for atom_idx, candidate in substituents:
            try:
                locant = seq.index(atom_idx) + 1
            except ValueError:
                entries = []
                break
            if locant <= 1:
                entries = []
                break
            entries.append((locant, atom_idx, candidate))
        if not entries:
            continue
        locants = tuple(sorted(locant for locant, _atom_idx, _candidate in entries))
        prefix_order = tuple(sorted((candidate.text, locant) for locant, _atom_idx, candidate in entries))
        options.append((locants, prefix_order, tuple(entries)))
    if not options:
        return None
    return min(options, key=lambda item: (item[0], item[1]))[2]


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


def _phenyl_disubstituent_locants(
    mol,
    ring_set: set[int],
    attachment_idx: int,
    substituent_atom_indices: list[int],
) -> tuple[int, int] | None:
    ring_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(attachment_idx).GetNeighbors()
        if nbr.GetIdx() in ring_set
    ]
    if len(ring_neighbors) != 2:
        return None
    options = []
    for first_idx in ring_neighbors:
        seq = _oriented_ring_sequence(mol, ring_set, attachment_idx, first_idx)
        if len(seq) != len(ring_set) or set(seq) != ring_set:
            continue
        try:
            locants = tuple(sorted(seq.index(atom_idx) + 1 for atom_idx in substituent_atom_indices))
        except ValueError:
            continue
        options.append(locants)
    return min(options) if options else None


def _oriented_ring_sequence(
    mol,
    ring_set: set[int],
    start_idx: int,
    first_idx: int,
) -> tuple[int, ...]:
    seq = [start_idx]
    previous = start_idx
    current = first_idx
    while len(seq) < len(ring_set):
        if current not in ring_set or current in seq:
            return ()
        seq.append(current)
        if len(seq) == len(ring_set):
            break
        next_indices = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(current).GetNeighbors()
            if nbr.GetIdx() in ring_set and nbr.GetIdx() != previous
        ]
        if len(next_indices) != 1:
            return ()
        previous, current = current, next_indices[0]
    return tuple(seq)


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


def _gem_difluoro_cycloalkyl_prefix(
    mol,
    ring_set: set[int],
    idx: int,
    parent_idx: int,
) -> PrefixCandidate | None:
    if len(ring_set) not in {4, 6}:
        return None
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in ring_set):
        return None
    external = _ring_external_heavy_neighbors_by_atom(mol, ring_set)
    fluorinated_atom = None
    fluorines: list[int] = []
    for atom_idx, neighbors in external.items():
        if atom_idx == idx:
            if neighbors != [parent_idx]:
                return None
            continue
        if not neighbors:
            continue
        if fluorinated_atom is not None:
            return None
        if len(neighbors) != 2:
            return None
        for neighbor_idx in neighbors:
            if not _is_terminal_fluorine(mol, neighbor_idx, atom_idx):
                return None
        fluorinated_atom = atom_idx
        fluorines = list(neighbors)
    if fluorinated_atom is None:
        return None
    locant = _cycloalkyl_substituent_locant(mol, ring_set, idx, fluorinated_atom)
    if locant is None or locant <= 1:
        return None
    parent_names = {4: "cyclobutyl", 6: "cyclohexyl"}
    return _prefix_candidate(
        f"{locant},{locant}-difluoro{parent_names[len(ring_set)]}",
        ring_set | set(fluorines),
        idx,
        parent_idx,
        "gem-difluoro-cycloalkyl-ring",
    )


def _is_terminal_fluorine(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "F" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return not any(nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1 for nbr in atom.GetNeighbors())


def _cycloalkyl_substituent_locant(
    mol,
    ring_set: set[int],
    attachment_idx: int,
    substituent_atom_idx: int,
) -> int | None:
    ring_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(attachment_idx).GetNeighbors()
        if nbr.GetIdx() in ring_set
    ]
    if len(ring_neighbors) != 2:
        return None
    locants = []
    for first_idx in ring_neighbors:
        seq = _oriented_ring_sequence(mol, ring_set, attachment_idx, first_idx)
        if len(seq) != len(ring_set) or set(seq) != ring_set:
            continue
        try:
            locants.append(seq.index(substituent_atom_idx) + 1)
        except ValueError:
            continue
    return min(locants) if locants else None


def _atom_has_assigned_stereo(atom) -> bool:
    return int(atom.GetChiralTag()) != 0


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
