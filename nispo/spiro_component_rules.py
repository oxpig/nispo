from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

from .branch_rules import branch_prefix
from .candidate_facts import (
    CandidateSafety,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,
)
from .cycle_utils import ordered_ring
from .engine_flags import FAST_POLY_BRIDGE_ROUTE_PROP
from .fused_template_families import (
    candidate_generated_fused_templates,
    curated_fused_templates,
)
from .formatting import format_prefixes as _format_prefixes
from .molecule_cache import mol_cache_dict, mol_cache_get
from .naming_context import MoleculeContext
from .parent.rule_helpers import (
    _adjust_fused_parent_for_charge,
    _has_assigned_bond_stereo,
    _iter_fused_template_matches,
    _partial_hydro_parent_names,
)
from .polycyclic_rules import (
    _unsaturation_contiguity_key,
    bicyclic_parent_candidate_records,
    kekulized_polycyclic_parent_candidate_records,
    spiro_parent_candidate_records,
)
from .ring_rules import ring_parent_candidates
from .ring_resource_rules import resource_ring_parent_candidates
from .ring_topology import ring_system_graph
from .rule_engine import NameCandidate
from .simple_prefixes import simple_external_prefix
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

_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT = 2
_SIMPLE_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT = 2
_COMPONENT_DISPIRO_TRIPLE_RENDER_LIMIT = 72
_COMPONENT_NAME_CACHE_LIMIT = 8192
_COMPONENT_NAME_CACHE: OrderedDict[tuple, tuple["_Component", ...]] = OrderedDict()


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(mol, idx, parent_idx)


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
        components_by_spiro = {
            idx: _components_without_atom(graph, idx) for idx in ring_system_atoms
        }
        spiro_split_count = sum(1 for parts in components_by_spiro.values() if len(parts) == 2)
        has_ring_system_carbonyl = None
        for spiro_idx in sorted(atom_set):
            parts = components_by_spiro[spiro_idx]
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
                if spiro_split_count != 1:
                    if has_ring_system_carbonyl is None:
                        has_ring_system_carbonyl = _has_ring_system_carbonyl(mol, Chem, ring_system_atoms)
                    if has_ring_system_carbonyl:
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
    rank = _cycle_rank_for_atoms(graph, ring_system_atoms)
    if rank > 4 and not (
        rank <= 7 and len(blocks) == 3
        or _is_high_rank_three_block_dispiro_chain(rank, blocks, edges)
    ):
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


def _is_high_rank_three_block_dispiro_chain(
    rank: int,
    blocks: Sequence[set[int]],
    edges: Sequence[tuple[int, int, int]],
) -> bool:
    if rank > 9 or len(blocks) != 3 or len(edges) != 2:
        return False
    spiro_atoms = {spiro_idx for _left_idx, _right_idx, spiro_idx in edges}
    if len(spiro_atoms) != 2:
        return False
    degrees = [0, 0, 0]
    for left_idx, right_idx, _spiro_idx in edges:
        degrees[left_idx] += 1
        degrees[right_idx] += 1
    return sorted(degrees) == [1, 1, 2]


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
    component_triples = sorted(
        (
            (_component_pre_render_key(left, center, right), left, center, right)
            for left in left_components[:6]
            for center in center_components[:6]
            if left_spiro in center.locants and right_spiro in center.locants
            for right in right_components[:6]
        ),
        key=lambda item: item[0],
    )
    for _triple_key, left, center, right in component_triples[:_COMPONENT_DISPIRO_TRIPLE_RENDER_LIMIT]:
        locants = {}
        locants.update(left.locants)
        locants.update({idx: _prime_locant(locant) for idx, locant in center.locants.items()})
        locants.update({idx: _double_prime_locant(locant) for idx, locant in right.locants.items()})
        covered_external_atoms = (
            set(left.covered_external_atoms)
            | set(center.covered_external_atoms)
            | set(right.covered_external_atoms)
        )
        substituent_prefixes = _external_substituent_prefixes(
            ctx.mol,
            ring_system_atoms,
            locants,
            covered_external_atoms=covered_external_atoms,
        )
        if substituent_prefixes is None:
            continue
        component_prefixes = []
        replacement_prefixes = []
        _extend_component_prefix_groups(component_prefixes, replacement_prefixes, left.replacement_prefixes)
        _extend_component_prefix_groups(
            component_prefixes,
            replacement_prefixes,
            center.replacement_prefixes,
            locant_transform=_prime_locant,
        )
        _extend_component_prefix_groups(
            component_prefixes,
            replacement_prefixes,
            right.replacement_prefixes,
            locant_transform=_double_prime_locant,
        )
        substituent_prefixes.extend(component_prefixes)
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
        render_rank = _component_spiro_render_rank(left, center, right)
        graph_atoms = frozenset(ring_system_atoms)
        candidates.append(
            NameCandidate(
                name=name,
                family="component-dispiro-parent",
                priority=priority,
                sort_key=(
                    left.component_rank + center.component_rank + right.component_rank,
                    len(replacement_prefixes),
                    (left.component_order, center.component_order, right.component_order),
                    (
                        left.numbering_key,
                        center.numbering_key,
                        right.numbering_key,
                    ),
                    _locant_key_for_text(name),
                    len(name),
                    name,
                ),
                stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, ctx.Chem, locants),
                stereo_locants=_component_spiro_stereo_locants(locants),
                parent_atoms=graph_atoms,
                ring_system_atoms=graph_atoms,
                safety=_combined_component_safety(left, center, right),
                render_cost=_combined_component_render_cost(left, center, right),
                render_rank=render_rank,
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
    component_atoms = set(atoms)
    expanded_atoms = _component_atoms_with_exocyclic_multiple_bond_atoms(ctx.mol, Chem, component_atoms)
    covered_external_atoms = frozenset(expanded_atoms - component_atoms)
    submol_data = _submol_for_atoms(ctx.mol, Chem, expanded_atoms)
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
    for sub_spiro in sub_spiros:
        raw_components.extend(_poly_component_names(submol, Chem, sub_spiro))

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
                component_order=component.component_order,
                numbering_key=component.numbering_key,
                covered_external_atoms=covered_external_atoms,
                safety=component.safety,
                render_cost=component.render_cost,
            )
        )
    components.sort(
        key=lambda item: (
            item.component_rank,
            len(item.replacement_prefixes),
            item.numbering_key,
            item.component_order,
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
    simple_expanded = _component_atoms_with_exocyclic_multiple_bond_atoms(ctx.mol, Chem, simple_atoms)
    poly_expanded = _component_atoms_with_exocyclic_multiple_bond_atoms(ctx.mol, Chem, poly_atoms)
    simple = _submol_for_atoms(ctx.mol, Chem, simple_expanded)
    poly = _submol_for_atoms(ctx.mol, Chem, poly_expanded)
    if simple is None or poly is None:
        return []
    simple_mol, simple_to_parent, simple_to_sub = simple
    poly_mol, poly_to_parent, poly_to_sub = poly
    simple_spiro = simple_to_sub[spiro_idx]
    poly_spiro = poly_to_sub[spiro_idx]

    simple_key = ("simple", tuple(sorted(simple_expanded)), spiro_idx)
    poly_key = ("poly", tuple(sorted(poly_expanded)), spiro_idx)
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
    component_pairs = sorted(
        (
            (_component_pre_render_key(simple_component, poly_component), simple_component, poly_component)
            for simple_component in simple_components[:4]
            for poly_component in poly_components[:12]
        ),
        key=lambda item: item[0],
    )
    for _pair_key, simple_component, poly_component in component_pairs[:_SIMPLE_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT]:
        component_prefixes = []
        parent_prefixes = []
        _extend_component_prefix_groups(component_prefixes, parent_prefixes, simple_component.replacement_prefixes)
        _extend_component_prefix_groups(
            component_prefixes,
            parent_prefixes,
            poly_component.replacement_prefixes,
            locant_transform=_prime_locant,
        )
        locants = _combined_locants(simple_component, simple_to_parent, poly_component, poly_to_parent)
        covered_external_atoms = (simple_expanded | poly_expanded) - ring_system_atoms
        substituent_prefixes = _external_substituent_prefixes(
            ctx.mol,
            ring_system_atoms,
            locants,
            covered_external_atoms=covered_external_atoms,
        )
        if substituent_prefixes is None:
            continue
        substituent_prefixes.extend(component_prefixes)
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
        render_rank = _component_spiro_render_rank(simple_component, poly_component)
        graph_atoms = frozenset(ring_system_atoms) if render_rank is not None else frozenset()
        candidates.append(
            NameCandidate(
                name=name,
                family="component-spiro-parent",
                priority=priority,
                sort_key=(
                    simple_component.component_rank + poly_component.component_rank,
                    len(parent_prefixes),
                    (simple_component.component_order, poly_component.component_order),
                    (simple_component.numbering_key, poly_component.numbering_key),
                    _locant_key_for_text(name),
                    len(name),
                    name,
                ),
                stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, Chem, locants),
                stereo_locants=_component_spiro_stereo_locants(locants),
                parent_atoms=graph_atoms,
                ring_system_atoms=graph_atoms,
                safety=_combined_component_safety(simple_component, poly_component),
                render_cost=_combined_component_render_cost(simple_component, poly_component),
                render_rank=render_rank,
            )
        )
        if _prefer_poly_first_component_spiro(poly_component):
            component_prefixes = []
            parent_prefixes = []
            _extend_component_prefix_groups(component_prefixes, parent_prefixes, poly_component.replacement_prefixes)
            _extend_component_prefix_groups(
                component_prefixes,
                parent_prefixes,
                simple_component.replacement_prefixes,
                locant_transform=_prime_locant,
            )
            locants = _combined_locants_poly_first(poly_component, poly_to_parent, simple_component, simple_to_parent)
            substituent_prefixes = _external_substituent_prefixes(
                ctx.mol,
                ring_system_atoms,
                locants,
                covered_external_atoms=covered_external_atoms,
            )
            if substituent_prefixes is None:
                continue
            substituent_prefixes.extend(component_prefixes)
            prefix_text = _component_spiro_prefix_text(substituent_prefixes, parent_prefixes)
            spiro_name = (
                "spiro["
                f"{_component_parent_text(poly_component.parent_name)}-{poly_component.spiro_locant},"
                f"{_prime_locant(simple_component.spiro_locant)}-{_component_parent_text(simple_component.parent_name)}"
                "]"
            )
            if _has_unpromoted_component_replacement(spiro_name):
                continue
            name = f"{prefix_text}-{spiro_name}" if prefix_text else spiro_name
            render_rank = 0
            graph_atoms = frozenset(ring_system_atoms)
            candidates.append(
                NameCandidate(
                    name=name,
                    family="component-spiro-parent",
                    priority=priority,
                    sort_key=(
                        simple_component.component_rank + poly_component.component_rank - 1,
                        len(parent_prefixes),
                        (poly_component.component_order, simple_component.component_order),
                        (poly_component.numbering_key, simple_component.numbering_key),
                        _locant_key_for_text(name),
                        len(name),
                        name,
                    ),
                    stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, Chem, locants),
                    stereo_locants=_component_spiro_stereo_locants(locants),
                    parent_atoms=graph_atoms,
                    ring_system_atoms=graph_atoms,
                    safety=_combined_component_safety(poly_component, simple_component),
                    render_cost=_combined_component_render_cost(poly_component, simple_component),
                    render_rank=render_rank,
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
    left_expanded = _component_atoms_with_exocyclic_multiple_bond_atoms(ctx.mol, Chem, left_atoms)
    right_expanded = _component_atoms_with_exocyclic_multiple_bond_atoms(ctx.mol, Chem, right_atoms)
    left = _submol_for_atoms(ctx.mol, Chem, left_expanded)
    right = _submol_for_atoms(ctx.mol, Chem, right_expanded)
    if left is None or right is None:
        return []
    left_mol, left_to_parent, left_to_sub = left
    right_mol, right_to_parent, right_to_sub = right
    left_spiro = left_to_sub[spiro_idx]
    right_spiro = right_to_sub[spiro_idx]

    left_key = ("poly", tuple(sorted(left_expanded)), spiro_idx)
    right_key = ("poly", tuple(sorted(right_expanded)), spiro_idx)
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
    component_pairs = sorted(
        (
            (_component_pre_render_key(left_component, right_component), left_component, right_component)
            for left_component in left_components[:12]
            for right_component in right_components[:12]
        ),
        key=lambda item: item[0],
    )
    for _pair_key, left_component, right_component in component_pairs[:_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT]:
        component_prefixes = []
        parent_prefixes = []
        _extend_component_prefix_groups(component_prefixes, parent_prefixes, left_component.replacement_prefixes)
        _extend_component_prefix_groups(
            component_prefixes,
            parent_prefixes,
            right_component.replacement_prefixes,
            locant_transform=_prime_locant,
        )
        locants = _combined_locants(left_component, left_to_parent, right_component, right_to_parent)
        covered_external_atoms = (left_expanded | right_expanded) - ring_system_atoms
        substituent_prefixes = _external_substituent_prefixes(
            ctx.mol,
            ring_system_atoms,
            locants,
            covered_external_atoms=covered_external_atoms,
        )
        if substituent_prefixes is None:
            continue
        substituent_prefixes.extend(component_prefixes)
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
        render_rank = _component_spiro_render_rank(left_component, right_component)
        graph_atoms = frozenset(ring_system_atoms) if render_rank is not None else frozenset()
        candidates.append(
            NameCandidate(
                name=name,
                family="poly-component-spiro-parent",
                priority=priority,
                sort_key=(
                    left_component.component_rank + right_component.component_rank,
                    len(parent_prefixes),
                    (left_component.component_order, right_component.component_order),
                    (left_component.numbering_key, right_component.numbering_key),
                    _locant_key_for_text(name),
                    len(name),
                    name,
                ),
                stereo_prefix=_component_spiro_stereo_prefix(ctx.mol, Chem, locants),
                stereo_locants=_component_spiro_stereo_locants(locants),
                parent_atoms=graph_atoms,
                ring_system_atoms=graph_atoms,
                safety=_combined_component_safety(left_component, right_component),
                render_cost=_combined_component_render_cost(left_component, right_component),
                render_rank=render_rank,
            )
        )
    return candidates


def _component_pre_render_key(*components: _Component) -> tuple:
    return (
        sum(component.component_rank for component in components),
        sum(len(component.replacement_prefixes) for component in components),
        tuple(component.component_order for component in components),
        tuple(component.numbering_key for component in components),
        tuple(_locant_key(component.spiro_locant) for component in components),
        sum(len(component.parent_name) for component in components),
        tuple(component.parent_name for component in components),
    )


def _extend_component_prefix_groups(
    component_prefixes: list[tuple[str, str]],
    replacement_prefixes: list[tuple[str, str]],
    prefixes: Sequence[tuple[str, str]],
    *,
    locant_transform=lambda locant: locant,
) -> None:
    for locant, prefix in prefixes:
        item = (locant_transform(locant), prefix)
        if _is_component_suffix_like_prefix(prefix):
            component_prefixes.append(item)
        else:
            replacement_prefixes.append(item)


def _is_component_suffix_like_prefix(prefix: str) -> bool:
    return prefix in {"oxo", "thioxo", "imino"}


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

    return _cached_component_spiro_prefix_text(
        tuple(substituent_prefixes),
        tuple(replacement_prefixes),
    )


@lru_cache(maxsize=32768)
def _cached_component_spiro_prefix_text(
    substituent_prefixes: tuple[tuple[str, str], ...],
    replacement_prefixes: tuple[tuple[str, str], ...],
) -> str:
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
        if not _is_component_stereo_prefix_locant(locant):
            return ""
        text = f"{locant}{descriptor}"
        if text in seen:
            continue
        seen.add(text)
        pieces.append((_locant_key(locant), text))
    return ",".join(text for _sort_key, text in sorted(pieces))


def _component_spiro_stereo_locants(locants: dict[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(
        (int(atom_idx), str(locant))
        for atom_idx, locant in locants.items()
        if _is_component_stereo_locant(str(locant))
    )


def _is_component_stereo_prefix_locant(locant: str) -> bool:
    base = str(locant).rstrip("'")
    return bool(base) and base.isdigit()


def _is_component_stereo_locant(locant: str) -> bool:
    text = str(locant)
    if text.endswith("''"):
        return False
    base = text[:-1] if text.endswith("'") else text
    return bool(base) and base.isdigit()


class _Component:
    def __init__(
        self,
        parent_name: str,
        spiro_locant: str,
        locants: dict[int, str],
        replacement_prefixes: tuple[tuple[str, str], ...] = (),
        component_rank: int = 0,
        component_kind: str = "",
        covered_external_atoms: frozenset[int] = frozenset(),
        component_order: int = 0,
        numbering_key: tuple = (0, ()),
        safety: CandidateSafety | None = None,
        render_cost: int = 1,
    ) -> None:
        self.parent_name = parent_name
        self.spiro_locant = spiro_locant
        self.locants = locants
        self.replacement_prefixes = replacement_prefixes
        self.component_rank = component_rank
        self.component_kind = component_kind
        self.covered_external_atoms = covered_external_atoms
        self.component_order = component_order
        self.numbering_key = numbering_key
        self.safety = safety
        self.render_cost = render_cost


def _component_name_cache_key(mol, Chem, family: str, spiro_idx: int) -> tuple:
    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetIsAromatic(),
            str(atom.GetChiralTag()),
            atom.GetNumExplicitHs(),
            atom.GetNoImplicit(),
            atom.GetNumRadicalElectrons(),
        )
        for atom in mol.GetAtoms()
    )
    bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                str(bond.GetBondType()),
                bond.GetIsAromatic(),
                str(bond.GetBondDir()),
                str(bond.GetStereo()),
                tuple(bond.GetStereoAtoms()),
            )
            for bond in mol.GetBonds()
        )
    )
    route_flag = (
        mol.HasProp(FAST_POLY_BRIDGE_ROUTE_PROP),
        mol.GetProp(FAST_POLY_BRIDGE_ROUTE_PROP) if mol.HasProp(FAST_POLY_BRIDGE_ROUTE_PROP) else "",
    )
    return (family, id(Chem), int(spiro_idx), route_flag, atoms, bonds)


def _cached_component_names(cache_key: tuple) -> list[_Component] | None:
    cached = _COMPONENT_NAME_CACHE.get(cache_key)
    if cached is None:
        return None
    _COMPONENT_NAME_CACHE.move_to_end(cache_key)
    return list(cached)


def _store_component_names(cache_key: tuple, components: list[_Component]) -> list[_Component]:
    _COMPONENT_NAME_CACHE[cache_key] = tuple(components)
    _COMPONENT_NAME_CACHE.move_to_end(cache_key)
    while len(_COMPONENT_NAME_CACHE) > _COMPONENT_NAME_CACHE_LIMIT:
        _COMPONENT_NAME_CACHE.popitem(last=False)
    return components


def _combined_component_safety(*components: _Component) -> CandidateSafety | None:
    reasons: list[str] = []
    for component in components:
        if component.safety is None:
            continue
        for reason in component.safety.opsin_fragility_reasons:
            if reason not in reasons:
                reasons.append(reason)
    if not reasons:
        return None
    return CandidateSafety(opsin_fragility_reasons=tuple(reasons))


def _combined_component_render_cost(*components: _Component) -> int:
    return 1 + sum(max(0, int(component.render_cost) - 1) for component in components)


def _simple_component_names(mol, Chem, spiro_idx: int) -> list[_Component]:
    cache_key = _component_name_cache_key(mol, Chem, "simple", spiro_idx)
    cached = _cached_component_names(cache_key)
    if cached is not None:
        return cached
    if mol.GetRingInfo().NumRings() != 1:
        return _store_component_names(cache_key, [])
    ring = tuple(mol.GetRingInfo().AtomRings()[0])
    ordered = ordered_ring(mol, ring)
    if not ordered:
        return _store_component_names(cache_key, [])
    records = list(ring_parent_candidates(mol, ordered))
    if not records:
        records = resource_ring_parent_candidates(mol, ordered, Chem)
    components = []
    for component_order, (parent_name, seq) in enumerate(records):
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
                component_kind="simple-ring",
                component_order=component_order,
            )
        )
    components.sort(
        key=lambda item: (
            len(item.replacement_prefixes),
            item.component_order,
            len(item.parent_name),
            item.parent_name,
            item.spiro_locant,
        )
    )
    return _store_component_names(cache_key, components)


def _poly_component_names(mol, Chem, spiro_idx: int) -> list[_Component]:
    cache_key = _component_name_cache_key(mol, Chem, "poly", spiro_idx)
    cached = _cached_component_names(cache_key)
    if cached is not None:
        return cached
    components: list[_Component] = []
    seen = set()
    record_groups = (
        (bicyclic_parent_candidate_records(mol, Chem), "generic-bicyclic"),
        (spiro_parent_candidate_records(mol, Chem), "generic-spiro"),
        (kekulized_polycyclic_parent_candidate_records(mol, Chem), "kekulized-polycyclic"),
    )
    component_order = 0
    for records, component_kind in record_groups:
        for record in records:
            raw_name, seq = record[0], record[1]
            bridge_unsaturation_mismatches = int(record[2]) if len(record) > 2 else 0
            if spiro_idx not in seq:
                component_order += 1
                continue
            parent_name, replacement_prefixes = _strip_leading_replacement_prefixes(raw_name)
            if not parent_name:
                component_order += 1
                continue
            locants = {atom_idx: str(index) for index, atom_idx in enumerate(seq, start=1)}
            key = (parent_name, tuple(replacement_prefixes), locants[spiro_idx])
            if key in seen:
                component_order += 1
                continue
            seen.add(key)
            component_rank = 3 if re.search(r"(?:^|-)\d*[a-z]*(?:bi|tri|tetra|penta)cyclo\[", parent_name) else 1
            safety = None
            render_cost = 1
            if bridge_unsaturation_mismatches > 0:
                safety = CandidateSafety(
                    opsin_fragility_reasons=(OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,),
                )
                render_cost = 1 + bridge_unsaturation_mismatches
            components.append(
                _Component(
                    parent_name=parent_name,
                    spiro_locant=locants[spiro_idx],
                    locants=locants,
                    replacement_prefixes=tuple(replacement_prefixes),
                    component_rank=component_rank,
                    component_kind=component_kind,
                    component_order=component_order,
                    numbering_key=_unsaturation_contiguity_key(mol, seq),
                    safety=safety,
                    render_cost=render_cost,
                )
            )
            component_order += 1
    components.sort(
        key=lambda item: (
            item.component_rank,
            len(item.replacement_prefixes),
            item.numbering_key,
            item.component_order,
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
            item.numbering_key,
            item.component_order,
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return _store_component_names(cache_key, components[:16])


@lru_cache(maxsize=8)
def _sorted_curated_fused_templates(Chem) -> tuple[dict, ...]:
    return tuple(
        sorted(
            curated_fused_templates(Chem),
            key=lambda item: (template_record_mol(item).GetNumAtoms(), template_record_name(item)),
        )
    )


def _fused_component_names(mol, Chem, spiro_idx: int, seen: set | None = None) -> list[_Component]:
    if mol.GetRingInfo().NumRings() < 2:
        return []
    seen = seen if seen is not None else set()
    ctx = MoleculeContext(mol, Chem)
    templates = (
        _sorted_curated_fused_templates(Chem)
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
    components.extend(_dihydroindene_spiro_components(components, seen))
    components.sort(
        key=lambda item: (
            _fused_component_parent_rank(item.parent_name),
            _locant_key(item.spiro_locant),
            len(item.parent_name),
            item.parent_name,
        )
    )
    return components[:12]


def _dihydroindene_spiro_components(components: Sequence[_Component], seen: set) -> list[_Component]:
    additions: list[_Component] = []
    for component in components:
        if component.parent_name not in {"indan", "indane"}:
            continue
        if component.spiro_locant != "2":
            continue
        parent_name = "1,3-dihydroindene"
        key = (parent_name, tuple(component.replacement_prefixes), component.spiro_locant)
        if key in seen:
            continue
        seen.add(key)
        additions.append(
            _Component(
                parent_name=parent_name,
                spiro_locant=component.spiro_locant,
                locants=component.locants,
                replacement_prefixes=component.replacement_prefixes,
                component_rank=-1,
                component_kind=component.component_kind,
                covered_external_atoms=component.covered_external_atoms,
                component_order=component.component_order,
                numbering_key=component.numbering_key,
                safety=component.safety,
                render_cost=component.render_cost,
            )
        )
    return additions


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
    if "hydro" in parent_name:
        return None
    return _Component(
        parent_name=parent_name,
        spiro_locant=spiro_locant,
        locants=locants,
        component_rank=0,
        component_kind="fused-template",
    )


_EXPLICIT_HETERO_GENERIC_COMPONENT_KINDS = frozenset(
    {
        "generic-bicyclic",
        "generic-spiro",
        "kekulized-polycyclic",
    }
)


def _component_spiro_render_rank(*components: _Component) -> int | None:
    """Rank component-spiro renderings using component provenance facts."""

    explicit_generic_replacements = sum(
        len(component.replacement_prefixes)
        for component in components
        if component.component_kind in _EXPLICIT_HETERO_GENERIC_COMPONENT_KINDS
    )
    if explicit_generic_replacements < 2:
        return None
    base_rank = (
        1
        + sum(component.component_rank for component in components) * 10
        + sum(len(component.replacement_prefixes) for component in components)
    )
    return base_rank * 10 + _component_spiro_orientation_rank(*components)


def _component_spiro_orientation_rank(*components: _Component) -> int:
    suffix_locants = []
    for component in components:
        for locant, term in component.replacement_prefixes:
            locant_number = _locant_key(locant)[0]
            if term in {"oxo", "thioxo", "imino"}:
                suffix_locants.append(locant_number)
    if suffix_locants:
        return min(9, min(suffix_locants))
    return 0


def _fused_component_parent_rank(name: str) -> tuple:
    generic = 1 if re.search(r"(?:^|-)\d*[a-z]*(?:bi|tri|tetra|penta)cyclo\[", name) else 0
    generated = 1 if "[" in name and re.search(r"\[[0-9,'a-z .-]+-[a-z]\]", name) else 0
    hydro = 1 if "hydro" in name else 0
    return (generic, generated, hydro)


def _strip_leading_replacement_prefixes(name: str) -> tuple[str, list[tuple[str, str]]]:
    prefixes: list[tuple[str, str]] = []
    rest = name.lstrip("-")
    prefix_terms = "|".join((*_REPLACEMENT_TERMS, "oxo", "thioxo", "imino"))
    while True:
        rest = rest.lstrip("-")
        match = re.match(
            rf"^(\d+(?:,\d+)*-)(?:di|tri|tetra|penta|hexa)?({prefix_terms})(.+)$",
            rest,
        )
        if match is None:
            break
        locant_text, replacement, rest = match.groups()
        for locant in locant_text[:-1].split(","):
            prefixes.append((locant, replacement))
    return rest.lstrip("-"), prefixes


def _component_parent_text(name: str) -> str:
    if re.match(r"^\d+(?:,\d+)+-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?hydro", name):
        return name
    return re.sub(r"^(\d+(?:,\d+)+)-([A-Za-z])", r"[\1]\2", name)


def _external_substituent_prefixes(
    mol,
    core_atoms: set[int],
    locants: dict[int, str],
    *,
    covered_external_atoms: set[int] | frozenset[int] = frozenset(),
) -> list[tuple[str, str]] | None:
    if mol_cache_get(mol, "candidate_telemetry") is None:
        cache = mol_cache_dict(mol, "spiro_external_substituent_prefixes")
        cache_key = (
            tuple(sorted(core_atoms)),
            tuple(sorted((int(idx), str(locant)) for idx, locant in locants.items())),
            tuple(sorted(int(idx) for idx in covered_external_atoms)),
            mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
            mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return list(cached)
    else:
        cache = None
        cache_key = None
    prefixes = []
    for atom_idx in sorted(core_atoms):
        locant = locants.get(atom_idx, "")
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in core_atoms:
                continue
            if nidx in covered_external_atoms:
                continue
            if not locant:
                return None
            prefix = _simple_or_branch_prefix(mol, nidx, atom_idx)
            if not prefix:
                return None
            prefixes.append((locant, prefix))
    if cache is not None and cache_key is not None:
        cache[cache_key] = tuple(prefixes)
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


def _combined_locants_poly_first(
    poly: _Component,
    poly_to_parent: dict[int, int],
    simple: _Component,
    simple_to_parent: dict[int, int],
) -> dict[int, str]:
    locants: dict[int, str] = {}
    for sub_idx, locant in poly.locants.items():
        locants[poly_to_parent[sub_idx]] = locant
    for sub_idx, locant in simple.locants.items():
        locants[simple_to_parent[sub_idx]] = _prime_locant(locant)
    return locants


def _prefer_poly_first_component_spiro(component: _Component) -> bool:
    parent = component.parent_name.lower()
    return parent in {"1,3-dihydroindene", "oxindol", "oxindole"} or parent.endswith("indol-2-one")


def _submol_for_atoms(mol, Chem, atoms: set[int]):
    ordered = sorted(atoms)
    parent_to_sub: dict[int, int] = {}
    rw = Chem.RWMol()
    for old_idx in ordered:
        source_atom = mol.GetAtomWithIdx(old_idx)
        atom = Chem.Atom(source_atom)
        removed_heavy_neighbors = [
            nbr
            for nbr in source_atom.GetNeighbors()
            if nbr.GetIdx() not in atoms and nbr.GetAtomicNum() > 1
        ]
        if removed_heavy_neighbors:
            atom.SetNoImplicit(False)
            if (
                source_atom.GetSymbol() == "N"
                and source_atom.GetFormalCharge() == 0
                and source_atom.GetIsAromatic()
            ):
                atom.SetNumExplicitHs(1)
            else:
                atom.SetNumExplicitHs(0)
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
    if mol.HasProp(FAST_POLY_BRIDGE_ROUTE_PROP):
        submol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, mol.GetProp(FAST_POLY_BRIDGE_ROUTE_PROP))
    sub_to_parent = {sub_idx: parent_idx for parent_idx, sub_idx in parent_to_sub.items()}
    return submol, sub_to_parent, parent_to_sub


def _component_atoms_with_exocyclic_multiple_bond_atoms(mol, Chem, atoms: set[int]) -> set[int]:
    expanded = set(atoms)
    for atom_idx in tuple(atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in expanded:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if nbr.GetSymbol() in {"O", "S", "N"}:
                if sum(1 for next_nbr in nbr.GetNeighbors() if next_nbr.GetAtomicNum() > 1) != 1:
                    continue
                expanded.add(nbr_idx)
    return expanded


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
    base = locant.rstrip("'")
    if base.isdigit():
        return (int(base), locant)
    digits = "".join(ch for ch in locant if ch.isdigit())
    return (int(digits) if digits else 999, locant)


def _locant_key_for_text(name: str) -> tuple[tuple[int, str], ...]:
    return tuple(_locant_key(locant) for locant in re.findall(r"(?<!\d)(\d+'?)(?=[,-])", name))


def _has_unpromoted_component_replacement(spiro_name: str) -> bool:
    replacement = "|".join(_REPLACEMENT_TERMS)
    return bool(re.search(rf"'\-\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?(?:{replacement})", spiro_name))
