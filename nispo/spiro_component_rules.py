from __future__ import annotations

import re
from collections.abc import Sequence

from .branch_rules import branch_prefix
from .cycle_utils import ordered_ring
from .formatting import format_prefixes as _format_prefixes
from .polycyclic_rules import (
    bicyclic_parent_candidate_records,
    kekulized_polycyclic_parent_candidate_records,
    spiro_parent_candidate_records,
)
from .ring_rules import ring_parent_candidates
from .ring_resource_rules import resource_ring_parent_candidates
from .ring_topology import ring_system_graph
from .rule_engine import NameCandidate


_REPLACEMENT_TERMS = (
    "tellura",
    "stanna",
    "selena",
    "phospha",
    "germa",
    "bora",
    "arsa",
    "sila",
    "thia",
    "oxa",
    "aza",
)


def component_spiro_parent_candidates(ctx, priority: int) -> list[NameCandidate]:
    """Compose spiro parents where one component is itself bicyclic/tricyclic.

    Generic spiro naming in :mod:`polycyclic_rules` handles simple spiro
    systems whose two sides are both single paths.  Patent-like benchmark
    failures often have a simple ring spiro-fused to a bridged or fused
    bicyclo/tricyclo component.  OPSIN accepts these as component names, for
    example ``spiro[cyclohexane-1,2'-tricyclo[...]]`` with hetero replacements
    and substituents locanted outside the bracket.  This rule builds that form
    from graph topology rather than from scaffold-specific templates.
    """

    mol = ctx.mol
    Chem = ctx.Chem
    candidates: list[NameCandidate] = []
    for atom_set in ctx.ring_systems:
        graph = ring_system_graph(mol, Chem, set(atom_set))
        if not graph:
            continue
        for spiro_idx in sorted(atom_set):
            parts = _components_without_atom(graph, spiro_idx)
            if len(parts) != 2:
                continue
            sides = []
            for part in parts:
                sub_atoms = set(part) | {spiro_idx}
                rank = _cycle_rank_for_atoms(graph, sub_atoms)
                sides.append((rank, sub_atoms))
            for simple_rank, simple_atoms, poly_rank, poly_atoms in _simple_poly_side_orders(sides):
                if simple_rank != 1 or poly_rank not in {2, 3, 4, 5}:
                    continue
                candidates.extend(
                    _component_spiro_candidates_for_split(
                        ctx,
                        set(atom_set),
                        spiro_idx,
                        simple_atoms,
                        poly_atoms,
                        priority,
                    )
                )
        candidates.extend(_component_dispiro_candidates(ctx, set(atom_set), graph, priority))
    candidates.sort(key=lambda item: item.sort_key)
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        unique.append(candidate)
        if len(unique) >= 24:
            break
    return unique


def _component_dispiro_candidates(
    ctx,
    ring_system_atoms: set[int],
    graph: dict[int, set[int]],
    priority: int,
) -> list[NameCandidate]:
    """Compose dispiro names from a chain of fused/simple ring components.

    Generic dispiro naming handles three simple rings.  Patent scaffolds often
    contain a simple middle ring connecting a simple terminal ring to a fused
    or bridged terminal component, which OPSIN parses as component-dispiro
    syntax.  Build that name from fused-ring blocks so this covers whole
    scaffold families rather than individual retained names.
    """

    blocks = _fused_ring_blocks(ctx.mol, ring_system_atoms)
    if len(blocks) < 3:
        return []
    edges: list[tuple[int, int, int]] = []
    for left_idx, left in enumerate(blocks):
        for right_idx in range(left_idx + 1, len(blocks)):
            shared = left & blocks[right_idx]
            if len(shared) == 1:
                edges.append((left_idx, right_idx, next(iter(shared))))
    if len(edges) < 2:
        return []
    adjacency: dict[int, list[tuple[int, int]]] = {idx: [] for idx in range(len(blocks))}
    for left_idx, right_idx, spiro_idx in edges:
        adjacency[left_idx].append((right_idx, spiro_idx))
        adjacency[right_idx].append((left_idx, spiro_idx))

    candidates: list[NameCandidate] = []
    for center_idx, links in adjacency.items():
        if len(links) < 2:
            continue
        for left_pos, (left_idx, left_spiro) in enumerate(links):
            for right_idx, right_spiro in links[left_pos + 1 :]:
                if left_spiro == right_spiro:
                    continue
                for terminal_left, terminal_right, spiro_left, spiro_right in (
                    (left_idx, right_idx, left_spiro, right_spiro),
                    (right_idx, left_idx, right_spiro, left_spiro),
                ):
                    candidates.extend(
                        _component_dispiro_candidates_for_blocks(
                            ctx,
                            ring_system_atoms,
                            blocks[terminal_left],
                            blocks[center_idx],
                            blocks[terminal_right],
                            spiro_left,
                            spiro_right,
                            priority,
                        )
                    )
    return candidates


def _component_dispiro_candidates_for_blocks(
    ctx,
    ring_system_atoms: set[int],
    left_atoms: set[int],
    center_atoms: set[int],
    right_atoms: set[int],
    left_spiro: int,
    right_spiro: int,
    priority: int,
) -> list[NameCandidate]:
    left_components = _mapped_component_names(ctx, left_atoms, (left_spiro,))
    center_components = _mapped_component_names(ctx, center_atoms, (left_spiro, right_spiro))
    right_components = _mapped_component_names(ctx, right_atoms, (right_spiro,))
    if not left_components or not center_components or not right_components:
        return []

    candidates: list[NameCandidate] = []
    for left in left_components[:6]:
        for center in center_components[:6]:
            if left_spiro not in center.locants or right_spiro not in center.locants:
                continue
            for right in right_components[:6]:
                locants = {}
                locants.update(left.locants)
                locants.update({idx: _prime_locant(locant) for idx, locant in center.locants.items()})
                locants.update({idx: _double_prime_locant(locant) for idx, locant in right.locants.items()})
                substituent_prefixes = _external_substituent_prefixes(ctx.mol, ring_system_atoms, locants)
                if substituent_prefixes is None:
                    continue
                replacement_prefixes = []
                replacement_prefixes.extend(left.replacement_prefixes)
                replacement_prefixes.extend(
                    (_prime_locant(locant), prefix) for locant, prefix in center.replacement_prefixes
                )
                replacement_prefixes.extend(
                    (_double_prime_locant(locant), prefix) for locant, prefix in right.replacement_prefixes
                )
                prefix_text = _component_spiro_prefix_text(substituent_prefixes, replacement_prefixes)
                dispiro_name = (
                    "dispiro["
                    f"{_component_parent_text(left.parent_name)}-{left.locants[left_spiro]},"
                    f"{_prime_locant(center.locants[left_spiro])}-{_component_parent_text(center.parent_name)}-"
                    f"{_prime_locant(center.locants[right_spiro])},"
                    f"{_double_prime_locant(right.locants[right_spiro])}-{_component_parent_text(right.parent_name)}"
                    "]"
                )
                if _has_unpromoted_component_replacement(dispiro_name):
                    continue
                name = f"{prefix_text}-{dispiro_name}" if prefix_text else dispiro_name
                candidates.append(
                    NameCandidate(
                        name=name,
                        family="component-dispiro-parent",
                        priority=priority,
                        sort_key=(
                            len(replacement_prefixes),
                            _locant_key_for_text(name),
                            len(name),
                            name,
                        ),
                    )
                )
    return candidates


def _mapped_component_names(ctx, atoms: set[int], spiro_atoms: Sequence[int]) -> list[_Component]:
    Chem = ctx.Chem
    submol_data = _submol_for_atoms(ctx.mol, Chem, atoms)
    if submol_data is None:
        return []
    submol, sub_to_parent, parent_to_sub = submol_data
    try:
        sub_spiros = tuple(parent_to_sub[idx] for idx in spiro_atoms)
    except KeyError:
        return []

    raw_components: list[_Component] = []
    if submol.GetRingInfo().NumRings() == 1:
        raw_components.extend(_simple_component_names(submol, Chem, sub_spiros[0]))
    if len(spiro_atoms) == 1:
        raw_components.extend(_poly_component_names(submol, Chem, sub_spiros[0]))

    components: list[_Component] = []
    seen = set()
    for component in raw_components:
        locants = {sub_to_parent[sub_idx]: locant for sub_idx, locant in component.locants.items()}
        if any(spiro_idx not in locants for spiro_idx in spiro_atoms):
            continue
        key = (
            component.parent_name,
            tuple(component.replacement_prefixes),
            tuple(locants[spiro_idx] for spiro_idx in spiro_atoms),
        )
        if key in seen:
            continue
        seen.add(key)
        components.append(
            _Component(
                parent_name=component.parent_name,
                spiro_locant=locants[spiro_atoms[0]],
                locants=locants,
                replacement_prefixes=component.replacement_prefixes,
            )
        )
    components.sort(
        key=lambda item: (
            len(item.replacement_prefixes),
            tuple(_locant_key(item.locants[idx]) for idx in spiro_atoms),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return components


def _fused_ring_blocks(mol, ring_system_atoms: set[int]) -> list[set[int]]:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= ring_system_atoms]
    blocks: list[set[int]] = []
    for ring in rings:
        merged = set(ring)
        changed = True
        while changed:
            changed = False
            remaining = []
            for block in blocks:
                if len(merged & block) > 1:
                    merged |= block
                    changed = True
                else:
                    remaining.append(block)
            blocks = remaining
        blocks.append(merged)
    return blocks


def _component_spiro_candidates_for_split(
    ctx,
    ring_system_atoms: set[int],
    spiro_idx: int,
    simple_atoms: set[int],
    poly_atoms: set[int],
    priority: int,
) -> list[NameCandidate]:
    Chem = ctx.Chem
    simple = _submol_for_atoms(ctx.mol, Chem, simple_atoms)
    poly = _submol_for_atoms(ctx.mol, Chem, poly_atoms)
    if simple is None or poly is None:
        return []
    simple_mol, simple_to_parent, simple_to_sub = simple
    poly_mol, poly_to_parent, poly_to_sub = poly
    simple_spiro = simple_to_sub[spiro_idx]
    poly_spiro = poly_to_sub[spiro_idx]

    simple_components = _simple_component_names(simple_mol, Chem, simple_spiro)
    poly_components = _poly_component_names(poly_mol, Chem, poly_spiro)
    if not simple_components or not poly_components:
        return []

    candidates: list[NameCandidate] = []
    for simple_component in simple_components[:4]:
        for poly_component in poly_components[:12]:
            parent_prefixes = list(simple_component.replacement_prefixes) + [
                (_prime_locant(locant), prefix)
                for locant, prefix in poly_component.replacement_prefixes
            ]
            locants = _combined_locants(simple_component, simple_to_parent, poly_component, poly_to_parent)
            substituent_prefixes = _external_substituent_prefixes(ctx.mol, ring_system_atoms, locants)
            if substituent_prefixes is None:
                continue
            prefix_text = _component_spiro_prefix_text(substituent_prefixes, parent_prefixes)
            spiro_name = (
                "spiro["
                f"{_component_parent_text(simple_component.parent_name)}-{simple_component.spiro_locant},"
                f"{_prime_locant(poly_component.spiro_locant)}-{_component_parent_text(poly_component.parent_name)}"
                "]"
            )
            if _has_unpromoted_component_replacement(spiro_name):
                continue
            name = f"{prefix_text}-{spiro_name}" if prefix_text else spiro_name
            candidates.append(
                NameCandidate(
                    name=name,
                    family="component-spiro-parent",
                    priority=priority,
                    sort_key=(
                        len(parent_prefixes),
                        _locant_key_for_text(name),
                        len(name),
                        name,
                    ),
                )
            )
    return candidates


def _component_spiro_prefix_text(
    substituent_prefixes: list[tuple[str, str]],
    replacement_prefixes: list[tuple[str, str]],
) -> str:
    """Format component-spiro prefixes in the order OPSIN tokenizes reliably.

    ``format_prefixes`` sorts prefix names alphabetically, which puts skeletal
    replacement terms such as ``aza`` before ordinary substituents.  OPSIN
    accepts component-spiro names much more consistently when substituents are
    applied before replacement prefixes, e.g.
    ``3'-ethoxycarbonyl-1',2',4'-triaza-spiro[...]``.
    """

    substituent_text = _format_prefixes(substituent_prefixes)
    replacement_text = _format_prefixes(replacement_prefixes)
    if substituent_text and replacement_text:
        return f"{substituent_text}-{replacement_text}"
    return substituent_text or replacement_text


class _Component:
    def __init__(
        self,
        parent_name: str,
        spiro_locant: str,
        locants: dict[int, str],
        replacement_prefixes: tuple[tuple[str, str], ...] = (),
    ) -> None:
        self.parent_name = parent_name
        self.spiro_locant = spiro_locant
        self.locants = locants
        self.replacement_prefixes = replacement_prefixes


def _simple_component_names(mol, Chem, spiro_idx: int) -> list[_Component]:
    if mol.GetRingInfo().NumRings() != 1:
        return []
    ring = tuple(mol.GetRingInfo().AtomRings()[0])
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return []
    records = list(ring_parent_candidates(mol, ordered))
    if not records:
        records = resource_ring_parent_candidates(mol, ordered, Chem)
    components = []
    for parent_name, seq in records:
        if spiro_idx not in seq:
            continue
        locants = {atom_idx: str(index) for index, atom_idx in enumerate(seq, start=1)}
        component_name, replacement_prefixes = _strip_leading_replacement_prefixes(parent_name)
        if component_name.startswith("cyclo"):
            parent_name = component_name
        else:
            replacement_prefixes = []
        components.append(
            _Component(
                parent_name=parent_name,
                spiro_locant=locants[spiro_idx],
                locants=locants,
                replacement_prefixes=tuple(replacement_prefixes),
            )
        )
    components.sort(
        key=lambda item: (
            len(item.replacement_prefixes),
            len(item.parent_name),
            item.parent_name,
            item.spiro_locant,
        )
    )
    return components


def _poly_component_names(mol, Chem, spiro_idx: int) -> list[_Component]:
    records = []
    records.extend(bicyclic_parent_candidate_records(mol, Chem))
    records.extend(spiro_parent_candidate_records(mol, Chem))
    records.extend(kekulized_polycyclic_parent_candidate_records(mol, Chem))
    components = []
    seen = set()
    for raw_name, seq in records:
        if spiro_idx not in seq:
            continue
        parent_name, replacement_prefixes = _strip_leading_replacement_prefixes(raw_name)
        if not parent_name:
            continue
        locants = {atom_idx: str(index) for index, atom_idx in enumerate(seq, start=1)}
        key = (parent_name, tuple(replacement_prefixes), locants[spiro_idx])
        if key in seen:
            continue
        seen.add(key)
        components.append(
            _Component(
                parent_name=parent_name,
                spiro_locant=locants[spiro_idx],
                locants=locants,
                replacement_prefixes=tuple(replacement_prefixes),
            )
        )
    components.sort(
        key=lambda item: (
            len(item.replacement_prefixes),
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return components


def _strip_leading_replacement_prefixes(name: str) -> tuple[str, list[tuple[str, str]]]:
    prefixes: list[tuple[str, str]] = []
    rest = name.lstrip("-")
    while True:
        match = re.match(
            rf"^(\d+(?:,\d+)*-)(?:di|tri|tetra|penta|hexa)?({'|'.join(_REPLACEMENT_TERMS)})(.+)$",
            rest,
        )
        if match is None:
            break
        locant_text, replacement, rest = match.groups()
        for locant in locant_text[:-1].split(","):
            prefixes.append((locant, replacement))
    return rest.lstrip("-"), prefixes


def _component_parent_text(name: str) -> str:
    return re.sub(r"^(\d+(?:,\d+)+)-([A-Za-z])", r"[\1]\2", name)


def _external_substituent_prefixes(
    mol,
    core_atoms: set[int],
    locants: dict[int, str],
) -> list[tuple[str, str]] | None:
    prefixes = []
    for atom_idx in sorted(core_atoms):
        locant = locants.get(atom_idx, "")
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms:
                continue
            if not locant:
                return None
            prefix = branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                return None
            prefixes.append((locant, prefix))
    return prefixes


def _combined_locants(
    simple: _Component,
    simple_to_parent: dict[int, int],
    poly: _Component,
    poly_to_parent: dict[int, int],
) -> dict[int, str]:
    locants: dict[int, str] = {}
    for sub_idx, locant in simple.locants.items():
        locants[simple_to_parent[sub_idx]] = locant
    for sub_idx, locant in poly.locants.items():
        locants[poly_to_parent[sub_idx]] = _prime_locant(locant)
    return locants


def _submol_for_atoms(mol, Chem, atoms: set[int]):
    ordered = sorted(atoms)
    parent_to_sub: dict[int, int] = {}
    rw = Chem.RWMol()
    for old_idx in ordered:
        atom = Chem.Atom(mol.GetAtomWithIdx(old_idx))
        parent_to_sub[old_idx] = rw.AddAtom(atom)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin not in parent_to_sub or end not in parent_to_sub:
            continue
        rw.AddBond(parent_to_sub[begin], parent_to_sub[end], bond.GetBondType())
        new_bond = rw.GetBondBetweenAtoms(parent_to_sub[begin], parent_to_sub[end])
        new_bond.SetIsAromatic(bond.GetIsAromatic())
        new_bond.SetStereo(bond.GetStereo())
    submol = rw.GetMol()
    try:
        Chem.SanitizeMol(submol)
    except Exception:
        return None
    sub_to_parent = {sub_idx: parent_idx for parent_idx, sub_idx in parent_to_sub.items()}
    return submol, sub_to_parent, parent_to_sub


def _components_without_atom(graph: dict[int, set[int]], atom_idx: int) -> list[set[int]]:
    remaining = set(graph) - {atom_idx}
    components = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            for nbr in graph[node]:
                if nbr != atom_idx and nbr not in component:
                    stack.append(nbr)
        remaining -= component
        components.append(component)
    return components


def _cycle_rank_for_atoms(graph: dict[int, set[int]], atoms: set[int]) -> int:
    edge_count = 0
    for atom_idx in atoms:
        edge_count += sum(1 for nbr in graph.get(atom_idx, ()) if nbr in atoms)
    return edge_count // 2 - len(atoms) + 1


def _simple_poly_side_orders(sides: Sequence[tuple[int, set[int]]]):
    if len(sides) != 2:
        return ()
    left, right = sides
    return (
        (left[0], left[1], right[0], right[1]),
        (right[0], right[1], left[0], left[1]),
    )


def _prime_locant(locant: str) -> str:
    return locant if locant.endswith("'") else locant + "'"


def _double_prime_locant(locant: str) -> str:
    return locant if locant.endswith("''") else locant.rstrip("'") + "''"


def _locant_key(locant: str) -> tuple[int, str]:
    digits = "".join(ch for ch in locant if ch.isdigit())
    return (int(digits) if digits else 999, locant)


def _locant_key_for_text(name: str) -> tuple[tuple[int, str], ...]:
    return tuple(_locant_key(locant) for locant in re.findall(r"(?<!\d)(\d+'?)(?=[,-])", name))


def _has_unpromoted_component_replacement(spiro_name: str) -> bool:
    replacement = "|".join(_REPLACEMENT_TERMS)
    return bool(re.search(rf"'\-\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?(?:{replacement})", spiro_name))
