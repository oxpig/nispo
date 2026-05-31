from __future__ import annotations

import re
from collections.abc import Sequence

from .branch_rules import branch_prefix
from .cycle_utils import ordered_ring
from .fused_template_families import (
    candidate_generated_fused_templates,
    curated_fused_templates,
)
from .formatting import format_prefixes as _format_prefixes
from .naming_context import MoleculeContext
from .parent.rule_helpers import (
    _adjust_fused_parent_for_charge,
    _has_assigned_bond_stereo,
    _iter_fused_template_matches,
    _partial_hydro_parent_names,
)
from .polycyclic_rules import (
    bicyclic_parent_candidate_records,
    kekulized_polycyclic_parent_candidate_records,
    spiro_parent_candidate_records,
)
from .ring_rules import ring_parent_candidates
from .ring_resource_rules import resource_ring_parent_candidates
from .ring_topology import ring_system_graph
from .rule_engine import NameCandidate
from .stereochemistry import assigned_atom_stereochemistry
from .template_access import template_record_locants, template_record_mol, template_record_name


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
    component_cache: dict[tuple, list[_Component]] = {}
    for atom_set in ctx.ring_systems:
        ring_system_atoms = set(atom_set)
        graph = ring_system_graph(mol, Chem, ring_system_atoms)
        if not graph:
            continue
        if _cycle_rank_for_atoms(graph, ring_system_atoms) > 7 and _has_ring_peroxide(mol, Chem, ring_system_atoms):
            continue
        spiro_split_count = sum(
            1 for idx in ring_system_atoms if len(_components_without_atom(graph, idx)) == 2
        )
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
                        ring_system_atoms,
                        spiro_idx,
                        simple_atoms,
                        poly_atoms,
                        priority,
                        component_cache,
                    )
                )
            for left_rank, left_atoms, right_rank, right_atoms in _poly_poly_side_orders(sides):
                if spiro_split_count != 1 and _has_ring_system_carbonyl(mol, Chem, ring_system_atoms):
                    continue
                if left_rank not in {2, 3, 4, 5} or right_rank not in {2, 3, 4, 5}:
                    continue
                candidates.extend(
                    _poly_component_spiro_candidates_for_split(
                        ctx,
                        ring_system_atoms,
                        spiro_idx,
                        left_atoms,
                        right_atoms,
                        priority,
                        component_cache,
                    )
                )
        candidates.extend(_component_dispiro_candidates(ctx, ring_system_atoms, graph, priority, component_cache))
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
    component_cache: dict[tuple, list[_Component]],
) -> list[NameCandidate]:
    """Compose dispiro names from a chain of fused/simple ring components.

    Generic dispiro naming handles three simple rings.  Patent scaffolds often
    contain a simple middle ring connecting a simple terminal ring to a fused
    or bridged terminal component, which OPSIN parses as component-dispiro
    syntax.  Build that name from fused-ring blocks so this covers whole
    scaffold families rather than individual retained names.
    """

    if _cycle_rank_for_atoms(graph, ring_system_atoms) > 4:
        return []

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
                            component_cache,
                        )
                    )
    return candidates


def _has_ring_peroxide(mol, Chem, atoms: set[int]) -> bool:
    return any(
        bond.GetBondType() == Chem.BondType.SINGLE
        and bond.GetBeginAtomIdx() in atoms
        and bond.GetEndAtomIdx() in atoms
        and bond.GetBeginAtom().GetSymbol() == "O"
        and bond.GetEndAtom().GetSymbol() == "O"
        for bond in mol.GetBonds()
    )


def _has_ring_system_carbonyl(mol, Chem, atoms: set[int]) -> bool:
    for atom_idx in atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C":
            continue
        if any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        ):
            return True
    return False


def _component_dispiro_candidates_for_blocks(
    ctx,
    ring_system_atoms: set[int],
    left_atoms: set[int],
    center_atoms: set[int],
    right_atoms: set[int],
    left_spiro: int,
    right_spiro: int,
    priority: int,
    component_cache: dict[tuple, list[_Component]],
) -> list[NameCandidate]:
    left_components = _mapped_component_names(ctx, left_atoms, (left_spiro,), component_cache)
    center_components = _mapped_component_names(ctx, center_atoms, (left_spiro, right_spiro), component_cache)
    right_components = _mapped_component_names(ctx, right_atoms, (right_spiro,), component_cache)
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
                            left.component_rank + center.component_rank + right.component_rank,
                            len(replacement_prefixes),
                            _locant_key_for_text(name),
                            len(name),
                            name,
                        ),
                        stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, ctx.Chem, locants),
                    )
                )
    return candidates


def _mapped_component_names(
    ctx,
    atoms: set[int],
    spiro_atoms: Sequence[int],
    component_cache: dict[tuple, list[_Component]] | None = None,
) -> list[_Component]:
    cache_key = ("mapped", tuple(sorted(atoms)), tuple(spiro_atoms))
    if component_cache is not None and cache_key in component_cache:
        return component_cache[cache_key]
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
                component_rank=component.component_rank,
            )
        )
    components.sort(
        key=lambda item: (
            item.component_rank,
            len(item.replacement_prefixes),
            tuple(_locant_key(item.locants[idx]) for idx in spiro_atoms),
            len(item.parent_name),
            item.parent_name,
        )
    )
    if component_cache is not None:
        component_cache[cache_key] = components
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
    component_cache: dict[tuple, list[_Component]] | None = None,
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

    simple_key = ("simple", tuple(sorted(simple_atoms)), spiro_idx)
    poly_key = ("poly", tuple(sorted(poly_atoms)), spiro_idx)
    if component_cache is not None and simple_key in component_cache:
        simple_components = component_cache[simple_key]
    else:
        simple_components = _simple_component_names(simple_mol, Chem, simple_spiro)
        if component_cache is not None:
            component_cache[simple_key] = simple_components
    if component_cache is not None and poly_key in component_cache:
        poly_components = component_cache[poly_key]
    else:
        poly_components = _poly_component_names(poly_mol, Chem, poly_spiro)
        if component_cache is not None:
            component_cache[poly_key] = poly_components
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
                        simple_component.component_rank + poly_component.component_rank,
                        len(parent_prefixes),
                        _locant_key_for_text(name),
                        len(name),
                        name,
                    ),
                    stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, Chem, locants),
                )
            )
    return candidates


def _poly_component_spiro_candidates_for_split(
    ctx,
    ring_system_atoms: set[int],
    spiro_idx: int,
    left_atoms: set[int],
    right_atoms: set[int],
    priority: int,
    component_cache: dict[tuple, list[_Component]] | None = None,
) -> list[NameCandidate]:
    """Compose spiro names where both sides are named ring components."""

    Chem = ctx.Chem
    left = _submol_for_atoms(ctx.mol, Chem, left_atoms)
    right = _submol_for_atoms(ctx.mol, Chem, right_atoms)
    if left is None or right is None:
        return []
    left_mol, left_to_parent, left_to_sub = left
    right_mol, right_to_parent, right_to_sub = right
    left_spiro = left_to_sub[spiro_idx]
    right_spiro = right_to_sub[spiro_idx]

    left_key = ("poly", tuple(sorted(left_atoms)), spiro_idx)
    right_key = ("poly", tuple(sorted(right_atoms)), spiro_idx)
    if component_cache is not None and left_key in component_cache:
        left_components = component_cache[left_key]
    else:
        left_components = _poly_component_names(left_mol, Chem, left_spiro)
        if component_cache is not None:
            component_cache[left_key] = left_components
    if component_cache is not None and right_key in component_cache:
        right_components = component_cache[right_key]
    else:
        right_components = _poly_component_names(right_mol, Chem, right_spiro)
        if component_cache is not None:
            component_cache[right_key] = right_components
    if not left_components or not right_components:
        return []

    candidates: list[NameCandidate] = []
    for left_component in left_components[:12]:
        for right_component in right_components[:12]:
            parent_prefixes = list(left_component.replacement_prefixes) + [
                (_prime_locant(locant), prefix)
                for locant, prefix in right_component.replacement_prefixes
            ]
            locants = _combined_locants(left_component, left_to_parent, right_component, right_to_parent)
            substituent_prefixes = _external_substituent_prefixes(ctx.mol, ring_system_atoms, locants)
            if substituent_prefixes is None:
                continue
            prefix_text = _component_spiro_prefix_text(substituent_prefixes, parent_prefixes)
            spiro_name = (
                "spiro["
                f"{_component_parent_text(left_component.parent_name)}-{left_component.spiro_locant},"
                f"{_prime_locant(right_component.spiro_locant)}-{_component_parent_text(right_component.parent_name)}"
                "]"
            )
            if _has_unpromoted_component_replacement(spiro_name):
                continue
            name = f"{prefix_text}-{spiro_name}" if prefix_text else spiro_name
            candidates.append(
                NameCandidate(
                    name=name,
                    family="poly-component-spiro-parent",
                    priority=priority,
                    sort_key=(
                        left_component.component_rank + right_component.component_rank,
                        len(parent_prefixes),
                        _locant_key_for_text(name),
                        len(name),
                        name,
                    ),
                    stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, Chem, locants),
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


def _component_spiro_stereo_prefix(mol, Chem, locants: dict[int, str]) -> str:
    if _has_assigned_bond_stereo(mol, Chem):
        return ""
    centers = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
    if not centers:
        return ""
    pieces = []
    seen = set()
    for idx, descriptor in centers:
        locant = locants.get(idx, "")
        if not re.match(r"^\d+'*$", locant):
            return ""
        text = f"{locant}{descriptor}"
        if text in seen:
            continue
        seen.add(text)
        pieces.append((_locant_key(locant), text))
    return ",".join(text for _sort_key, text in sorted(pieces))


class _Component:
    def __init__(
        self,
        parent_name: str,
        spiro_locant: str,
        locants: dict[int, str],
        replacement_prefixes: tuple[tuple[str, str], ...] = (),
        component_rank: int = 0,
    ) -> None:
        self.parent_name = parent_name
        self.spiro_locant = spiro_locant
        self.locants = locants
        self.replacement_prefixes = replacement_prefixes
        self.component_rank = component_rank


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
    components: list[_Component] = []
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
            item.component_rank,
            len(item.replacement_prefixes),
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    components.extend(_fused_component_names(mol, Chem, spiro_idx, seen))
    components.sort(
        key=lambda item: (
            item.component_rank,
            len(item.replacement_prefixes),
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return components[:16]


def _fused_component_names(mol, Chem, spiro_idx: int, seen: set | None = None) -> list[_Component]:
    if mol.GetRingInfo().NumRings() < 2:
        return []
    seen = seen if seen is not None else set()
    ctx = MoleculeContext(mol, Chem)
    templates = (
        tuple(
            sorted(
                curated_fused_templates(Chem),
                key=lambda item: (template_record_mol(item).GetNumAtoms(), template_record_name(item)),
            )
        )
        + candidate_generated_fused_templates(mol, Chem, ctx.features)
    )
    components: list[_Component] = []
    for template, match in _iter_fused_template_matches(
        ctx,
        templates,
        Chem,
        reject_extra_core_bonds=True,
    ):
        component = _component_from_fused_template(mol, template, match, spiro_idx)
        if component is None:
            continue
        key = (
            component.parent_name,
            tuple(component.replacement_prefixes),
            component.spiro_locant,
        )
        if key in seen:
            continue
        seen.add(key)
        components.append(component)
    for template, match in _iter_fused_template_matches(
        ctx,
        templates,
        Chem,
        topology_query=True,
        require_one_ring_system=True,
    ):
        if len(match) != mol.GetNumAtoms() or set(match) != set(range(mol.GetNumAtoms())):
            continue
        for parent_name, hydro_key in _partial_hydro_parent_names(mol, template, match):
            component = _component_from_fused_template(mol, template, match, spiro_idx, parent_name=parent_name)
            if component is None:
                continue
            key = (
                component.parent_name,
                tuple(component.replacement_prefixes),
                component.spiro_locant,
            )
            if key in seen:
                continue
            seen.add(key)
            components.append(component)
            if len(components) >= 12:
                break
    components.sort(
        key=lambda item: (
            _fused_component_parent_rank(item.parent_name),
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return components[:12]


def _component_from_fused_template(
    mol,
    template: dict,
    match: Sequence[int],
    spiro_idx: int,
    *,
    parent_name: str | None = None,
) -> _Component | None:
    if len(match) != mol.GetNumAtoms() or set(match) != set(range(mol.GetNumAtoms())):
        return None
    if spiro_idx not in match:
        return None
    reverse = {mol_idx: template_idx for template_idx, mol_idx in enumerate(match)}
    template_idx = reverse[spiro_idx]
    locants = {
        mol_idx: str(template_record_locants(template).get(template_idx, ""))
        for template_idx, mol_idx in enumerate(match)
    }
    locants = {idx: locant for idx, locant in locants.items() if locant}
    spiro_locant = locants.get(spiro_idx, "")
    if not spiro_locant or spiro_locant.endswith("a"):
        return None
    parent_name = parent_name or _adjust_fused_parent_for_charge(mol, template, reverse)
    if not parent_name:
        return None
    return _Component(parent_name=parent_name, spiro_locant=spiro_locant, locants=locants, component_rank=1)


def _fused_component_parent_rank(name: str) -> tuple:
    generic = 1 if re.search(r"(?:^|-)\d*[a-z]*(?:bi|tri|tetra|penta)cyclo\[", name) else 0
    generated = 1 if "[" in name and re.search(r"\[[0-9,'a-z .-]+-[a-z]\]", name) else 0
    hydro = 1 if "hydro" in name else 0
    return (generic, generated, hydro)


def _strip_leading_replacement_prefixes(name: str) -> tuple[str, list[tuple[str, str]]]:
    prefixes: list[tuple[str, str]] = []
    rest = name.lstrip("-")
    while True:
        rest = rest.lstrip("-")
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


def _poly_poly_side_orders(sides: Sequence[tuple[int, set[int]]]):
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
