from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

from .molecule_iteration import iter_atoms, iter_bonds
from .branch_rules import branch_prefix
from .candidate_facts import (
    CandidateSafety,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,
)
from .cycle_utils import ordered_ring
from .engine_flags import FAST_POLY_BRIDGE_ROUTE_PROP
from .fused_template_families import (
    curated_fused_templates,
)
from .constants import MULT
from .formatting import format_prefixes as _format_prefixes, join_prefix_parent
from .molecule_cache import mol_cache_dict, mol_cache_get
from .ring_parent_selection import component_spiro_parent_facts
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

_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT = 1
_SIMPLE_POLY_COMPONENT_SPIRO_PAIR_RENDER_LIMIT = 1
_COMPONENT_DISPIRO_TRIPLE_RENDER_LIMIT = 72
_POLY_COMPONENT_NAME_RECORD_LIMIT = 3
_COMPONENT_NAME_CACHE_LIMIT = 8192
_COMPONENT_NAME_CACHE: OrderedDict[tuple, tuple["_Component", ...]] = OrderedDict()
_UNSTABLE_COMPONENT_SPIRO_RETAINED_PARENT_NAMES = frozenset(
    {
        "1H-pyrido[2,3-b][1,4]oxazin-2(3H)-one",
        "[1,3]benzodioxane",
        "[1,4]benzoxazine",
        "isoindolin-1-one",
    }
)
_UNSTABLE_COMPONENT_SPIRO_RETAINED_PARENT_FRAGMENTS = (
    "benzodiox",
    "benzoxazin",
    "dihydro-cyclopenta[a]phenanthren",
    "dihydro-safro",
    "pyrazolo[4,3-c]pyridin",
    "safro",
    "[1,3]benzodioxane",
    "[1,4]benzoxazine",
    "pyrido[2,3-b][1,4]oxazin",
    "isoindolin-1-one",
)


def _simple_or_branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return simple_external_prefix(mol, idx, parent_idx) or branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=True,
    )


def component_spiro_parent_candidates(ctx, priority: int, *, attachment=None,
                                     branch_names=None) -> list[NameCandidate]:
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
        if attachment is not None and (
                attachment[0] not in ring_system_atoms or attachment[1] in ring_system_atoms):
            continue
        graph = ring_system_graph(mol, Chem, ring_system_atoms)
        if not graph:
            continue
        if _cycle_rank_for_atoms(graph, ring_system_atoms) > 7 and _has_ring_peroxide(mol, Chem, ring_system_atoms):
            continue
        components_by_spiro = _two_component_splits_by_articulation(graph, ring_system_atoms)
        spiro_split_count = len(components_by_spiro)
        has_ring_system_carbonyl = None
        for spiro_idx in sorted(components_by_spiro):
            parts = components_by_spiro[spiro_idx]
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
                        attachment,
                        branch_names,
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
                        attachment,
                        branch_names,
                    )
                )
        candidates.extend(_component_dispiro_candidates(
            ctx, ring_system_atoms, graph, priority, component_cache,
            attachment=attachment, branch_names=branch_names,
        ))
    # A nested spiro component has two independent priming levels. The
    # two-component renderer does not preserve that locant scope. Where the
    # complete chain has a direct dispiro construction, retain its explicit
    # three-component mapping instead of emitting the ambiguous nested form.
    dispiro_scopes = {c.parent_atoms for c in candidates
                     if c.family == 'component-dispiro-parent'}
    candidates = [c for c in candidates
                  if not (c.parent_atoms in dispiro_scopes
                          and c.family in {'component-spiro-parent', 'poly-component-spiro-parent'}
                          and c.name.count('spiro[') > 1)]
    candidates.sort(key=lambda item: item.sort_key)
    seen = set()
    unique = []
    for candidate in candidates:
        marker = candidate.name if attachment is None else (
            candidate.name, dict(candidate.stereo_locants).get(attachment[0]))
        if marker in seen:
            continue
        seen.add(marker)
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
    *,
    attachment=None,
    branch_names=None,
) -> list[NameCandidate]:
    """Compose dispiro names from a chain of fused/simple ring components.

    Generic dispiro naming handles three simple rings.  Patent scaffolds often
    contain a simple middle ring connecting a simple terminal ring to a fused
    or bridged terminal component, which OPSIN parses as component-dispiro
    syntax.  Build that name from fused-ring blocks so this covers whole
    scaffold families rather than individual retained names.
    """

    blocks = _fused_ring_blocks(ctx.mol, ring_system_atoms)
    if len(blocks) != 3 or set().union(*blocks) != ring_system_atoms:
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
                            attachment=attachment,
                            branch_names=branch_names,
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
        for bond in iter_bonds(mol)
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


def _dispiro_component_hydride(component):
    """Decode a proved component operation before composing its dispiro parent."""
    parent = component.parent_name
    hydro = ()
    match = re.match(r"^([0-9a-z,]+)-([a-z]+hydro)-?(.+)$", parent)
    if match is not None:
        locations, prefix, base = match.groups()
        locants = tuple(locations.split(","))
        if prefix == MULT.get(len(locants), "") + "hydro" and set(locants) <= set(component.locants.values()):
            parent, hydro = base, locants
    indicated = ()
    match = re.match(r"^([0-9]+[a-z]?H(?:,[0-9]+[a-z]?H)*)-(.+)$", parent)
    if match is not None:
        labels, base = match.groups()
        locants = tuple(label[:-1] for label in labels.split(","))
        if set(locants) <= set(component.locants.values()):
            parent, indicated = base, locants
    return parent, hydro, indicated


def _dispiro_component_citation(parent):
    # Ignore intrinsic numeric locants when comparing distinct terminal hydrides.
    return re.sub(r"^(?:[0-9,]+-|\[[0-9,]+\])", "", parent).lower()


def _dispiro_component_text(parent):
    if re.fullmatch(r"[12]-benzofuran", parent):
        return f"[{parent[0]}]benzofuran"
    return _component_parent_text(parent)


def _proved_neutral_dispiro_junctions(mol, components, blocks, spiro_atoms):
    """Prove two neutral carbon junctions on three numbered component hydrides."""
    hydride_kinds = {"mapped-ring-hydride", "simple-ring", "fused-template"}
    if any(component.component_kind not in hydride_kinds
           or set(component.locants) != block
           or len(set(component.locants.values())) != len(block)
           for component, block in zip(components, blocks)):
        return False
    for offset, idx in enumerate(spiro_atoms):
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetTotalNumHs() or atom.GetDegree() != 4
                or any(bond.GetBondTypeAsDouble() != 1 for bond in atom.GetBonds())):
            return False
        neighbors = {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
        left_neighbors = neighbors & blocks[offset]
        right_neighbors = neighbors & blocks[offset + 1]
        if len(left_neighbors) != 2 or len(right_neighbors) != 2 or left_neighbors & right_neighbors:
            return False
    return True


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
    *,
    attachment=None,
    branch_names=None,
) -> list[NameCandidate]:
    left_components = _mapped_component_names(ctx, left_atoms, (left_spiro,), component_cache)
    center_components = _mapped_component_names(ctx, center_atoms, (left_spiro, right_spiro), component_cache)
    right_components = _mapped_component_names(ctx, right_atoms, (right_spiro,), component_cache)
    if not left_components or not center_components or not right_components:
        return []

    # P-31.2.3: retain an available numbered heterocyclic component rather
    # than replacing its heteroatoms on a carbon skeleton to change citation
    # order. Apply this only to complete, annotation-safe mapped hydrides.
    def component_hydrides(components):
        native = [component for component in components
                  if not any(prefix in _REPLACEMENT_TERMS
                             for _locant, prefix in component.replacement_prefixes)
                  and component.safety is None]
        return native or components

    left_components = component_hydrides(left_components)
    center_components = component_hydrides(center_components)
    right_components = component_hydrides(right_components)

    candidates: list[NameCandidate] = []
    component_triples = sorted(
        (
            (_component_pre_render_key(left, center, right), left, center, right)
            for left in left_components[:6]
            for center in center_components[:6]
            if left_spiro in center.locants and right_spiro in center.locants
            for right in right_components[:6]
            if _dispiro_component_citation(_dispiro_component_hydride(left)[0])
               <= _dispiro_component_citation(_dispiro_component_hydride(right)[0])
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
            attachment=attachment,
            branch_names=branch_names,
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
        junctions = (left.locants[left_spiro], _prime_locant(center.locants[left_spiro]),
                     _prime_locant(center.locants[right_spiro]), _double_prime_locant(right.locants[right_spiro]))
        hydrides = tuple(map(_dispiro_component_hydride, (left, center, right)))
        if all(hydro for _parent, hydro, _indicated in hydrides):
            # Grouped hydro on all three independently hydrogenated components
            # does not have a graph-preserving interoperability route. Preserve
            # the numbered central hydride and remap the two terminal operations.
            hydrides = (hydrides[0], (center.parent_name, (), ()), hydrides[2])
        bases = tuple(_dispiro_component_text(parent) for parent, _hydro, _indicated in hydrides)
        dispiro_name = f"dispiro[{bases[0]}-{junctions[0]},{junctions[1]}-{bases[1]}-{junctions[2]},{junctions[3]}-{bases[2]}]"
        if _has_unpromoted_component_replacement(dispiro_name):
            continue
        transforms = (str, _prime_locant, _double_prime_locant)
        hydro_sites = sorted((transform(locant)
                              for transform, (_, hydro, _) in zip(transforms, hydrides)
                              for locant in hydro), key=_locant_key)
        indicated_sites = sorted({transform(locant)
                                  for transform, (_, _, indicated) in zip(transforms, hydrides)
                                  for locant in indicated}, key=_locant_key)
        if indicated_sites:
            dispiro_name = ",".join(locant + "H" for locant in indicated_sites) + "-" + dispiro_name
        if hydro_sites:
            multiplier = MULT.get(len(hydro_sites))
            if multiplier is None:
                continue
            dispiro_name = (",".join(hydro_sites) + "-" + multiplier + "hydro"
                            + ("-" if indicated_sites else "") + dispiro_name)
        name = join_prefix_parent(prefix_text, dispiro_name)
        if attachment is None:
            promoted = _complete_spiro_carbonyl_name(
                ctx.mol, ring_system_atoms, locants, substituent_prefixes,
                replacement_prefixes, dispiro_name,
            )
            if promoted:
                name = promoted
        graph_atoms = frozenset(ring_system_atoms)
        safety = _combined_component_safety(left, center, right)
        render_rank = _component_spiro_render_rank(left, center, right)
        if safety is None and _proved_neutral_dispiro_junctions(
            ctx.mol, (left, center, right), (left_atoms, center_atoms, right_atoms),
            (left_spiro, right_spiro),
        ):
            render_rank = 0
        candidates.append(
            NameCandidate(
                name=name,
                family="component-dispiro-parent",
                parent_selection=component_spiro_parent_facts(ctx.mol, ring_system_atoms, locants, dispiro_name, name),
                priority=priority,
                sort_key=(
                    left.component_rank + center.component_rank + right.component_rank,
                    len(replacement_prefixes),
                    (left.component_order, center.component_order, right.component_order),
                    tuple(sorted(map(_locant_key, junctions))),
                    tuple(map(_locant_key, junctions)),
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
                safety=safety,
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
                component_kind=component.component_kind,
                component_order=component.component_order,
                numbering_key=component.numbering_key,
                covered_external_atoms=covered_external_atoms,
                safety=component.safety,
                render_cost=component.render_cost,
                junction_proved=component.junction_proved,
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
    attachment=None,
    branch_names=None,
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

    # The capped cycle leaves two equivalent directions after fixing the
    # junction. Number its actual substituents before limiting the pair pool.
    first = simple_components[0]
    if attachment is not None and first.junction_proved and not first.replacement_prefixes:
        first = min((item for item in simple_components if item.junction_proved
                     and item.parent_name == first.parent_name and not item.replacement_prefixes
                     and item.numbering_key == first.numbering_key),
                    key=lambda item: _locant_key(item.spiro_locant))
    if first.junction_proved and not first.replacement_prefixes:
        directions = [
            item for item in simple_components
            if item.junction_proved and item.parent_name == first.parent_name
            and item.spiro_locant == first.spiro_locant
            and item.numbering_key == first.numbering_key and not item.replacement_prefixes
        ]
        if directions:
            sites = [
                simple_to_sub[idx] for idx in simple_atoms
                for atom in ctx.mol.GetAtomWithIdx(idx).GetNeighbors()
                if atom.GetAtomicNum() > 1 and atom.GetIdx() not in simple_expanded
                and atom.GetIdx() not in ring_system_atoms
                and (idx, atom.GetIdx()) != attachment
            ]
            root = simple_to_sub.get(attachment[0]) if attachment is not None else None
            chosen = min(directions, key=lambda item: (
                _locant_key(item.locants[root]) if root is not None else (),
                tuple(sorted(_locant_key(item.locants[idx]) for idx in sites))))
            if chosen is not simple_components[0]:
                simple_components = [chosen, *(item for item in simple_components if item is not chosen)]

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
            attachment=attachment,
            branch_names=branch_names,
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
        name = prefix_text + spiro_name
        promoted = None if attachment is not None else _complete_spiro_carbonyl_name(
            ctx.mol, ring_system_atoms, locants, substituent_prefixes,
            parent_prefixes, spiro_name,
        )
        if promoted:
            name = promoted
        render_rank = _component_spiro_render_rank(simple_component, poly_component)
        graph_atoms = frozenset(ring_system_atoms) if render_rank is not None else frozenset()
        candidates.append(
            NameCandidate(
                name=name,
                family="component-spiro-parent",
                parent_selection=component_spiro_parent_facts(ctx.mol, ring_system_atoms, locants, spiro_name, name),
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
                attachment=attachment,
                branch_names=branch_names,
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
            name = prefix_text + spiro_name
            promoted = None if attachment is not None else _complete_spiro_carbonyl_name(
                ctx.mol, ring_system_atoms, locants, substituent_prefixes,
                parent_prefixes, spiro_name,
            )
            if promoted:
                name = promoted
            render_rank = 0
            graph_atoms = frozenset(ring_system_atoms)
            candidates.append(
                NameCandidate(
                    name=name,
                    family="component-spiro-parent",
                    parent_selection=component_spiro_parent_facts(ctx.mol, ring_system_atoms, locants, spiro_name, name),
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
    attachment=None,
    branch_names=None,
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
            attachment=attachment,
            branch_names=branch_names,
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
        name = prefix_text + spiro_name
        promoted = None if attachment is not None else _complete_spiro_carbonyl_name(
            ctx.mol, ring_system_atoms, locants, substituent_prefixes,
            parent_prefixes, spiro_name,
        )
        if promoted:
            name = promoted
        render_rank = _component_spiro_render_rank(left_component, right_component)
        graph_atoms = frozenset(ring_system_atoms) if render_rank is not None else frozenset()
        candidates.append(
            NameCandidate(
                name=name,
                family="poly-component-spiro-parent",
                parent_selection=component_spiro_parent_facts(ctx.mol, ring_system_atoms, locants, spiro_name, name),
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
    return bool(re.fullmatch(r"\d+[a-z]?'*", str(locant)))


def _is_component_stereo_locant(locant: str) -> bool:
    # Fused-component bridgeheads and every primed component retain their
    # original atom identity in the assembled spiro parent.
    return _is_component_stereo_prefix_locant(locant)


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
        junction_proved: bool = False,
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
        self.junction_proved = junction_proved


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
        for atom in iter_atoms(mol)
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
            for bond in iter_bonds(mol)
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
    # A saturated ring with an exocyclic carbonyl still has the saturated
    # parent hydride. Name its capped ring first, then restore that terminal
    # multiple-bond group as an owned component prefix below.
    ring_atoms = frozenset(mol.GetRingInfo().AtomRings()[0])
    if all(b.GetBondTypeAsDouble() == 1 for b in mol.GetBonds()
           if b.GetBeginAtomIdx() in ring_atoms and b.GetEndAtomIdx() in ring_atoms):
        mapped = _proved_hydride_component_names(mol, Chem, spiro_idx)
        if mapped:
            return _store_component_names(cache_key, mapped)
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
        # The expanded fragment contains terminal multiple-bond atoms. Ring
        # hydride recognition accounts only for the cycle; explicitly retain
        # each exocyclic group before the assembly marks those atoms covered.
        for owner in seq:
            for neighbor in mol.GetAtomWithIdx(owner).GetNeighbors():
                if neighbor.GetIdx() in locants:
                    continue
                bond = mol.GetBondBetweenAtoms(owner, neighbor.GetIdx())
                term = {8: "oxo", 16: "thioxo", 7: "imino"}.get(neighbor.GetAtomicNum())
                if (term and neighbor.GetDegree() == 1
                        and bond.GetBondTypeAsDouble() == 2):
                    replacement_prefixes.append((locants[owner], term))
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
    mapped = _proved_hydride_component_names(mol, Chem, spiro_idx)
    if mapped:
        return _store_component_names(cache_key, mapped)
    components: list[_Component] = []
    seen = set()
    record_groups = (
        (bicyclic_parent_candidate_records(mol, Chem), "generic-bicyclic"),
        (spiro_parent_candidate_records(mol, Chem), "generic-spiro"),
        (
            kekulized_polycyclic_parent_candidate_records(
                mol,
                Chem,
                limit=_POLY_COMPONENT_NAME_RECORD_LIMIT,
            ),
            "kekulized-polycyclic",
        ),
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
    if _should_try_retained_fused_component_names(components):
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


def _proved_hydride_component_names(mol, Chem, spiro_idx: int) -> list[_Component]:
    """Compose a hydride from its numbered topology and proved H population.

    The capped graph supplies vocabulary and maps only. Whole-assembly
    substituents, isotope/stereo descriptors and suffixes still use the
    caller's original atoms. No parser selects a component or a numbering.
    """
    from .mapped_ring_parents import _core_graph, skeletons
    from .ring_state_parent import _inventory, _reference_states, _render

    if mol.GetStereoGroups() or any(a.GetIsotope() or a.GetFormalCharge()
            or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return []
    core = frozenset(i for ring in mol.GetRingInfo().AtomRings() for i in ring)
    if spiro_idx not in core or len(core) > 24:
        return []
    junction = mol.GetAtomWithIdx(spiro_idx)
    neutral_carbon_junction = (
        junction.GetAtomicNum() == 6 and junction.GetDegree() == 2
        and junction.GetTotalNumHs() == 2
        and all(bond.GetBondTypeAsDouble() == 1 for bond in junction.GetBonds())
    )
    external = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in core:
            continue
        if (atom.GetAtomicNum() not in {7, 8, 16} or atom.GetDegree() != 1
                or atom.GetNeighbors()[0].GetIdx() not in core
                or atom.GetBonds()[0].GetBondTypeAsDouble() != 2):
            return []
        external.append((atom.GetNeighbors()[0].GetIdx(),
                         {7: 'imino', 8: 'oxo', 16: 'thioxo'}[atom.GetAtomicNum()]))
    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return []
    graph, original = prepared
    Chem.RemoveStereochemistry(graph)
    ctx = MoleculeContext(graph, Chem)
    inventory = _inventory(graph, Chem)
    if not inventory.supported or inventory.groups:
        return []
    rows = []
    for skeleton in skeletons(graph, Chem, frozenset(range(graph.GetNumAtoms()))):
        references = ((None,) if skeleton[5] or len(skeleton) > 8
                      else _reference_states(skeleton[3], skeleton[4]))
        for reference in references:
            rows.extend(_render(ctx, skeleton, reference, inventory))
    if not rows or len({row.parent for row in rows}) != 1:
        return []
    minimum = min(row.numbering_key for row in rows)
    components, seen = [], set()
    for row in rows:
        if row.numbering_key != minimum or row.prefixes:
            continue
        locants = {original[i]: loc for i, loc in row.atom_locants}
        spiro_locant = locants.get(spiro_idx, '')
        if not spiro_locant:
            continue
        component_name = row.name
        if spiro_locant in row.indicated_h_locants:
            # P-24.3.2: a neutral carbon junction replaces both hydrogens
            # of the capped methylene. Retain the separate hydro operation
            # on the proved ring and omit its now-consumed indicated H.
            construction = row.construction
            if (not neutral_carbon_junction or not row.hydrogen_budget_proved
                    or row.indicated_h_locants != (spiro_locant,) or row.added_h_locants
                    or construction is None or not construction.hydro_prefix
                    or construction.parent != f"{spiro_locant}H-{row.parent}"):
                continue
            from .formatting import join_prefix_parent
            component_name = join_prefix_parent(construction.hydro_prefix, row.parent)
        # Hantzsch-Widman locants are intrinsic to that vocabulary. For
        # example, dioxolane is not an oxo prefix followed by "lane".
        replacement_parent = (row.method.startswith('skeletal-replacement-')
                              or row.method in {'von-Baeyer', 'spiro', 'dispiro'})
        parent, replacements = (_strip_leading_replacement_prefixes(component_name)
                                if replacement_parent else (component_name, []))
        replacements.extend((locants[owner], term) for owner, term in external)
        key = (parent, tuple(replacements), tuple(sorted(locants.items())))
        if key in seen:
            continue
        seen.add(key)
        components.append(_Component(parent, spiro_locant, locants,
            replacement_prefixes=tuple(replacements), component_rank=0,
            component_kind='mapped-ring-hydride', numbering_key=_unsaturation_contiguity_key(
                mol, tuple(original[i] for i in row.parent_atoms)),
            junction_proved=(neutral_carbon_junction and row.hydrogen_budget_proved
                             and row.annotation_coverage and set(locants) == core
                             and len(set(locants.values())) == len(core))))
    return components


def _should_try_retained_fused_component_names(components: Sequence[_Component]) -> bool:
    if not components:
        return True
    for component in components:
        if component.component_rank >= 3:
            return True
        if component.replacement_prefixes and component.component_kind in _EXPLICIT_HETERO_GENERIC_COMPONENT_KINDS:
            return True
        if component.safety is not None or component.render_cost > 1:
            return True
    return False


@lru_cache(maxsize=8)
def _sorted_curated_fused_templates(Chem) -> tuple[dict, ...]:
    return tuple(
        sorted(
            curated_fused_templates(Chem),
            key=lambda item: (template_record_mol(item).GetNumAtoms(), template_record_name(item)),
        )
    )


@lru_cache(maxsize=256)
def _curated_fused_component_templates_by_atom_count(Chem, atom_count: int) -> tuple[dict, ...]:
    return tuple(
        template
        for template in _sorted_curated_fused_templates(Chem)
        if template_record_mol(template).GetNumAtoms() == atom_count
    )


def _fused_component_names(mol, Chem, spiro_idx: int, seen: set | None = None) -> list[_Component]:
    if mol.GetRingInfo().NumRings() < 2:
        return []
    seen = seen if seen is not None else set()
    ctx = MoleculeContext(mol, Chem)
    templates = _curated_fused_component_templates_by_atom_count(Chem, mol.GetNumAtoms())
    if not templates:
        return []
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
    # A fused carbon with a lettered fusion locant is a junction of the
    # numbered parent. An ordinary carbon spiro center needs two bonds in
    # each component, so that locant cannot name this attachment.
    if not spiro_locant.isdecimal():
        return None
    exact_template = parent_name is None
    parent_name = parent_name or _adjust_fused_parent_for_charge(mol, template, reverse)
    if not parent_name:
        return None
    if _unstable_component_spiro_retained_parent_name(parent_name):
        return None
    # A capped indene junction replaces the two H atoms on C-1.
    # Its complete template match fixes the double bond and every locant.
    junction = mol.GetAtomWithIdx(spiro_idx)
    indene_junction = (
        exact_template and parent_name in {"inden", "indene", "1H-indene"}
        and spiro_locant == "1" and len(set(locants.values())) == len(match)
        and junction.GetAtomicNum() == 6 and junction.GetTotalNumHs() == 2
        and junction.GetDegree() == 2 and not junction.GetFormalCharge()
        and all(bond.GetBondTypeAsDouble() == 1 for bond in junction.GetBonds())
        and all(not a.GetFormalCharge() and not a.GetIsotope() and not a.GetChiralTag()
                for a in mol.GetAtoms()))
    return _Component(
        parent_name="indene" if indene_junction else parent_name,
        spiro_locant=spiro_locant,
        locants=locants,
        component_rank=0,
        component_kind="fused-template",
        junction_proved=indene_junction,
    )


def _unstable_component_spiro_retained_parent_name(parent_name: str) -> bool:
    if parent_name in _UNSTABLE_COMPONENT_SPIRO_RETAINED_PARENT_NAMES:
        return True
    return any(fragment in parent_name for fragment in _UNSTABLE_COMPONENT_SPIRO_RETAINED_PARENT_FRAGMENTS)


_EXPLICIT_HETERO_GENERIC_COMPONENT_KINDS = frozenset(
    {
        "generic-bicyclic",
        "generic-spiro",
        "kekulized-polycyclic",
        "mapped-ring-hydride",
    }
)


def _component_spiro_render_rank(*components: _Component) -> int | None:
    """Rank component-spiro renderings using component provenance facts."""
    if all(component.junction_proved for component in components):
        return 0

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
    attachment=None,
    branch_names=None,
) -> list[tuple[str, str]] | None:
    if branch_names is None and mol_cache_get(mol, "candidate_telemetry") is None:
        cache = mol_cache_dict(mol, "spiro_external_substituent_prefixes")
        cache_key = (
            tuple(sorted(core_atoms)),
            tuple(sorted((int(idx), str(locant)) for idx, locant in locants.items())),
            tuple(sorted(int(idx) for idx in covered_external_atoms)),
            attachment,
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
            if attachment == (atom_idx, nidx):
                continue
            if nidx in covered_external_atoms:
                continue
            if not locant:
                return None
            prefix = (_simple_or_branch_prefix(mol, nidx, atom_idx) if branch_names is None
                      else branch_names.get((atom_idx, nidx), ""))
            if not prefix:
                return None
            prefixes.append((locant, prefix))
    if cache is not None and cache_key is not None:
        cache[cache_key] = tuple(prefixes)
    return prefixes


def mapped_component_spiro_yl(mol, Chem, root, host, scope, *, allow_replacement=False,
                             general_branches=False):
    """One original free valence on a numbered spiro assembly."""
    result = _mapped_component_spiro(
        mol, Chem, root, host, scope, "yl",
        allow_replacement=allow_replacement, general_branches=general_branches)
    if result is None:
        return "", frozenset()
    name, locants = result
    from .component_mapping import _record_branch_ring_atoms

    _record_branch_ring_atoms(mol, Chem, root, host, name, scope, dict(locants))
    return name, scope


def _mapped_component_spiro(mol, Chem, root, host, scope, suffix, *,
                           allow_replacement=False, general_branches=False):
    if len(scope) < 8 or mol.GetRingInfo().NumRings() < 3 or mol.GetStereoGroups():
        return None
    from .mapped_ring_parents import ring_systems

    core = next((system for system in ring_systems(mol) if root in system), frozenset())
    if not core <= scope or host in core or not any(
            sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 4 for i in core):
        return None
    atom = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, host)
    if (cut is None or cut.GetBondTypeAsDouble() != 1 or atom.GetAtomicNum() not in {6, 7}
            or int(cut.GetStereo())):
        return None
    if general_branches:
        if (atom.GetAtomicNum() == 7 and atom.GetTotalValence() != 3
                or sum(n.GetIdx() in core for n in atom.GetNeighbors()) != 2):
            return None
    elif atom.GetIsAromatic() or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds()):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in scope):
        return None
    branch_names = None
    if general_branches:
        branch_names = _mapped_spiro_branches(mol, Chem, core, scope, (root, host))
        if branch_names is None:
            return None
    else:
        for i in scope - core:
            peripheral = mol.GetAtomWithIdx(i)
            if (peripheral.GetDegree() != 1 or peripheral.GetNeighbors()[0].GetIdx() not in core
                    or peripheral.GetAtomicNum() not in {6, 8, 9, 17, 35, 53}):
                return None
    if {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in scope} != {(root, host)}:
        return None

    from .component_mapping import MappedComponent, decorate_component
    from .formatting import attach_suffix

    ctx = MoleculeContext(mol, Chem)
    candidates = component_spiro_parent_candidates(ctx, 0, attachment=(root, host), branch_names=branch_names)
    # Identical unrooted words can use different locants for the free
    # valence. Compare those locants before rendering the rooted word.
    eligible = []
    for item in candidates:
        if (item.parent_atoms != core or item.render_rank is None
                or item.render_rank != 0 and not allow_replacement or item.safety is not None):
            continue
        locants = dict(item.stereo_locants)
        if locants.keys() != core or len(set(locants.values())) != len(core):
            continue
        eligible.append((item, locants))
    eligible.sort(key=lambda row: (row[0].sort_key, _locant_key(row[1][root])))
    for item, locants in eligible:
        parent = attach_suffix(item.name, suffix, (locants[root],))
        name = decorate_component(mol, Chem, MappedComponent(
            parent, tuple(locants.items()), core, embedded_atoms=scope - core,
            systematic_parent=True,
        ))
        if name:
            return name, tuple(locants.items())
    return None


def _mapped_spiro_branches(mol, Chem, core, scope, attachment):
    from .component_mapping import mapped_general_branch

    names, owned = {}, set()
    for i in core:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j in core or (i, j) == attachment:
                continue
            atom = mol.GetAtomWithIdx(j)
            if atom.GetAtomicNum() <= 1 or j not in scope:
                return None
            if (bond.GetBondTypeAsDouble() == 2 and atom.GetDegree() == 1
                    and atom.GetAtomicNum() in {7, 8, 16} and not int(bond.GetStereo())):
                # The component hydride owns this terminal multiple bond.
                atoms = frozenset({j})
            else:
                text, atoms = mapped_general_branch(mol, Chem, j, i)
                if not text or any(c.isspace() for c in text):
                    return None
                names[i, j] = text
            if atoms & owned or not atoms <= scope - core:
                return None
            owned.update(atoms)
    return names if owned == scope - core else None


def mapped_spiro_carboxamide(mol, Chem, group, scope):
    """Complete a single exocyclic amide on its numbered spiro assembly."""
    owner, center, nitrogen = group.owner, group.center, group.ligand
    if owner < 0 or not mol.GetAtomWithIdx(owner).IsInRing():
        return None
    from .branch_rules import collect_subtree

    acid_side = frozenset(collect_subtree(mol, owner, center))
    parent = _mapped_component_spiro(
        mol, Chem, owner, center, acid_side, "carboxamide", general_branches=True)
    if parent is None or acid_side & set(group.atoms):
        return None
    parent, locants = parent
    from .component_mapping import mapped_general_branch
    from .formatting import format_prefixes

    prefixes, occupied = [], set(acid_side) | set(group.atoms)
    for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
        root = atom.GetIdx()
        if root == center or atom.GetAtomicNum() <= 1:
            continue
        text, atoms = mapped_general_branch(mol, Chem, root, nitrogen)
        if not text or atoms & occupied or any(c.isspace() for c in text):
            return None
        prefixes.append(("N", text))
        occupied |= atoms
    if occupied != scope:
        return None
    prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
    # The ring's own descriptor precedes the combined prefix sequence.
    stereo = re.match(r"^\([^()]+\)-", parent)
    if stereo is not None:
        name = stereo.group() + join_prefix_parent(prefix, parent[stereo.end():])
    else:
        name = join_prefix_parent(prefix, parent)
    return name, locants


def mapped_spiro_ester(mol, Chem, arm, scope):
    """Join the mapped ester organyl to an original numbered spiro parent."""
    from .branch_rules import collect_subtree
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation

    group = arm.group
    acid_side = frozenset(collect_subtree(mol, group.owner, group.center))
    if (acid_side & (set(group.atoms) | arm.atoms)
            or acid_side | set(group.atoms) | arm.atoms != scope):
        return None
    parent = _mapped_component_spiro(mol, Chem, group.owner, group.center, acid_side,
                                    "carboxylate", general_branches=True)
    if parent is None:
        return None
    anion, locants = parent
    rule = 'P-24/P-65.6.3.3.1'
    return MethodDecision(f'{arm.name} {anion}', 'mapped-ester', frozenset({group.center}), scope,
        rule, locants, preference_issues=(SelectionLimitation(
            'ester-parent-and-method', rule, 'Complete ester parent comparison remains open.'),))


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
    # A spiro component is named as a hydride before the whole assembly
    # receives its substituent prefixes and principal suffix.
    submol.SetProp("_nispo_principal_prefix_mode", "1")
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


def _two_component_splits_by_articulation(
    graph: dict[int, set[int]],
    atoms: set[int],
) -> dict[int, list[set[int]]]:
    if not graph:
        return {}
    if not _graph_is_connected(graph, atoms):
        return {
            idx: parts
            for idx in atoms
            if len(parts := _components_without_atom(graph, idx)) == 2
        }
    return {
        idx: parts
        for idx in _articulation_atoms(graph, atoms)
        if len(parts := _components_without_atom(graph, idx)) == 2
    }


def _graph_is_connected(graph: dict[int, set[int]], atoms: set[int]) -> bool:
    if not atoms:
        return True
    start = next(iter(atoms))
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(nbr for nbr in graph.get(node, ()) if nbr in atoms and nbr not in seen)
    return seen == atoms


def _articulation_atoms(graph: dict[int, set[int]], atoms: set[int]) -> set[int]:
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    articulation: set[int] = set()
    time = 0

    def visit(node: int) -> None:
        nonlocal time
        time += 1
        discovery[node] = low[node] = time
        child_count = 0
        for nbr in graph.get(node, ()):
            if nbr not in atoms:
                continue
            if nbr not in discovery:
                parent[nbr] = node
                child_count += 1
                visit(nbr)
                low[node] = min(low[node], low[nbr])
                if parent.get(node) is None:
                    if child_count > 1:
                        articulation.add(node)
                elif low[nbr] >= discovery[node]:
                    articulation.add(node)
            elif nbr != parent.get(node):
                low[node] = min(low[node], discovery[nbr])

    for node in atoms:
        if node in discovery:
            continue
        parent[node] = None
        visit(node)
    return articulation


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


def _complete_spiro_carbonyl_name(mol, core_atoms, locants, prefixes, replacements, parent):
    """Apply acid or ketone suffixes to a complete, numbered assembly.

    Fragment hydrides are named before the whole assembly gets its suffix.
    Each promoted site must own its original mapped prefix on that parent.
    """
    from .principal_carbonyl import principal_carbonyl_sites
    from .principal_carboxyl import ring_carboxyl_suffix
    from .formatting import join_prefix_parent, multiplied_suffix

    if mol.HasProp("_nispo_principal_prefix_mode"):
        return ""
    acid = ring_carboxyl_suffix(mol, locants, prefixes, parent)
    if acid:
        ending, remaining, _center, _locant = acid
        return join_prefix_parent(_component_spiro_prefix_text(remaining, replacements), ending)
    suffix, sites = principal_carbonyl_sites(mol)
    if suffix != "one" or not sites or len(sites) > 6:
        return ""
    positions = []
    for carbonyl, owner in sites:
        locant = locants.get(owner, "")
        if (carbonyl != owner or owner not in core_atoms
                or not re.fullmatch(r"\d+'{0,2}", locant)
                or (locant, "oxo") not in prefixes):
            return ""
        positions.append(locant)
    removed = {(locant, "oxo") for locant in positions}
    remaining = [record for record in prefixes if record not in removed]
    text = _component_spiro_prefix_text(remaining, replacements)
    positions.sort(key=_locant_key)
    ending = multiplied_suffix(len(positions), "one") if len(positions) > 1 else "one"
    name = f"{parent}-{','.join(positions)}-{ending}"
    return join_prefix_parent(text, name)


@lru_cache(maxsize=1)
def spiro_sulfamate_query(Chem):
    return Chem.MolFromSmarts('[S;R;X4](=[O;D1])(=[O;D1])(-[O;R;X2])-[N;R;X3]')


def spiro_sulfamate_carbamate(mol, Chem):
    """Name a saturated monospiro cyclic sulfamate with an N-carbamate."""
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .formatting import attach_suffix, enclose_name_unit
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .polycyclic_rules import _ring_system_graph, _spiro_parent_sequence_candidates

    matches = mol.GetSubstructMatches(spiro_sulfamate_query(Chem))
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    if (len(matches) != 1 or len(rings) != 2 or len(rings[0] & rings[1]) != 1
            or mol.GetStereoGroups() or any(a.GetIsotope() or a.GetFormalCharge()
                or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms())):
        return None
    sulfur, oxo1, oxo2, oxygen, nitrogen = matches[0]
    core = rings[0] | rings[1]
    if (not any({sulfur, oxygen, nitrogen} <= ring for ring in rings)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core - {sulfur, oxygen, nitrogen})
            or any(b.GetBondTypeAsDouble() != 1 and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= core
                   for b in mol.GetBonds())):
        return None
    attachments = [(i, n.GetIdx()) for i in core for n in mol.GetAtomWithIdx(i).GetNeighbors()
                   if n.GetIdx() not in core | {oxo1, oxo2}]
    if len(attachments) != 1 or attachments[0][0] != nitrogen:
        return None
    carbonyl = mol.GetAtomWithIdx(attachments[0][1])
    if carbonyl.GetAtomicNum() != 6 or carbonyl.GetDegree() != 3:
        return None
    ligands = {b.GetBondTypeAsDouble(): b.GetOtherAtom(carbonyl) for b in carbonyl.GetBonds()
               if b.GetOtherAtomIdx(carbonyl.GetIdx()) != nitrogen}
    if set(ligands) != {1, 2} or any(a.GetAtomicNum() != 8 for a in ligands.values()):
        return None
    alcohol, oxo = ligands[1], ligands[2]
    if alcohol.GetDegree() != 2 or oxo.GetDegree() != 1:
        return None
    root = next(a.GetIdx() for a in alcohol.GetNeighbors() if a.GetIdx() != carbonyl.GetIdx())
    word, atoms = mapped_general_branch(mol, Chem, root, alcohol.GetIdx())
    functional = frozenset({carbonyl.GetIdx(), alcohol.GetIdx(), oxo.GetIdx(), oxo1, oxo2})
    scope = frozenset(range(mol.GetNumAtoms()))
    if (not word or atoms & (core | functional) or atoms | core | functional != scope
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    candidates = []
    for parent, sequence in _spiro_parent_sequence_candidates(mol, core, _ring_system_graph(mol, Chem, core)):
        locants = {idx: i for i, idx in enumerate(sequence, 1)}
        key = (tuple(sorted(locants[i] for i in (oxygen, sulfur, nitrogen))),
               tuple(locants[i] for i in (oxygen, sulfur, nitrogen)))
        spiro = parent.find('spiro[')
        if spiro < 0:
            continue
        replacements = '-'.join(f'{locants[idx]}-{term}' for idx, term in
                                ((oxygen, 'oxa'), (sulfur, 'thia'), (nitrogen, 'aza')))
        hydride = attach_suffix(replacements + parent[spiro:], 'carboxylate', (str(locants[nitrogen]),))
        mapping = tuple((idx, str(loc)) for idx, loc in locants.items())
        # The two sulfur oxo groups share a mapped locant. Their prefix keeps
        # the valence explicit without requiring an unlocanted spiro fusion.
        component = MappedComponent(hydride, mapping, core,
            ((str(locants[sulfur]), 'oxo'),) * 2, scope - core, systematic_parent=True)
        text = decorate_component(mol, Chem, component)
        if text:
            candidates.append((key, text, mapping))
    if not candidates:
        return None
    _, text, mapping = min(candidates)
    organyl = enclose_name_unit(word) if word[0].isdigit() else word
    rule = 'P-24.2.4/P-65.6/P-67.1.2.4'
    return MethodDecision(organyl + ' ' + text, 'functional-class',
        frozenset({carbonyl.GetIdx()}), scope, rule, mapping,
        preference_issues=(SelectionLimitation('spiro-sulfamate', rule,
            'The spiro junction, sulfamate heteroatoms and N-carbamate are mapped.'),))


def numbered_thia_spiro_dioxide(mol, Chem):
    """Name a monocyclic spiro pair with both oxo groups on a ring sulfur."""
    from .extended_inventory import selection_inventory
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .polycyclic_rules import _spiro_ring_system_candidate_records

    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    atoms = frozenset(range(mol.GetNumAtoms()))
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    if not inventory.supported or inventory.principal:
        return None
    cores = set()
    for sulfur in mol.GetAtoms():
        if sulfur.GetAtomicNum() != 16 or not sulfur.IsInRing():
            continue
        center = sulfur.GetIdx()
        oxo = {b.GetOtherAtomIdx(center) for b in sulfur.GetBonds()
               if b.GetBondTypeAsDouble() == 2
               and b.GetOtherAtom(sulfur).GetAtomicNum() == 8
               and b.GetOtherAtom(sulfur).GetDegree() == 1}
        if len(oxo) != 2:
            continue
        for left in (ring for ring in rings if center in ring):
            for right in rings:
                if len(left & right) != 1:
                    continue
                core = left | right
                if sum(bool(core & ring) for ring in rings) != 2 or any(
                    b.GetBondTypeAsDouble() != 1 for i in core
                    for b in mol.GetAtomWithIdx(i).GetBonds() if b.GetOtherAtomIdx(i) in core
                ):
                    continue
                cores.add(core)
    if len(cores) != 1:
        return None
    candidates = _spiro_ring_system_candidate_records(mol, Chem, cores.pop(), max_records=1, render_late=True)
    if not candidates:
        return None
    name, order = candidates[0]
    rule = 'P-24.2.4/P-64.4'
    mapping = tuple((i, str(locant)) for locant, i in enumerate(order, 1))
    return MethodDecision(name, 'substitutive', frozenset(), atoms, rule, mapping,
        preference_issues=(SelectionLimitation('spiro-parent', rule,
            'Both sulfur oxo groups use the same numbered spiro framework.'),))


def sulfonyl_monospiro_ketone(mol, Chem, inventory=None):
    """Promote a ring ketone on a saturated monoazaspiro sulfonamide."""
    from .component_mapping import mapped_general_branch
    from .extended_inventory import selection_inventory
    from .formatting import attach_suffix, join_prefix_parent, format_prefixes
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .polycyclic_rules import _ring_system_graph, _spiro_parent_sequence_candidates

    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    if len(rings) != 2 or len(rings[0] & rings[1]) != 1 or mol.GetStereoGroups():
        return None
    core = rings[0] | rings[1]
    if any(a.GetIsotope() or a.GetFormalCharge() or a.GetChiralTag() or a.GetNumRadicalElectrons()
           for a in mol.GetAtoms()):
        return None
    if any(b.GetBondTypeAsDouble() != 1 and
           {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= core for b in mol.GetBonds()):
        return None
    inventory = inventory if inventory is not None else selection_inventory(mol)
    if not inventory.supported or len(inventory.principal) != 1:
        return None
    sulfonamide = inventory.principal[0]
    nitrogen = sulfonamide.ligand
    ketones = tuple(group for group in inventory.groups if group.kind == "ketone" and group.center in core)
    if (sulfonamide.kind != "sulfonamide" or nitrogen not in core or len(ketones) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != (7 if i == nitrogen else 6) for i in core)):
        return None
    ketone = ketones[0]
    text, branch = mapped_general_branch(mol, Chem, sulfonamide.center, nitrogen)
    if (not text or branch & core or branch & set(ketone.atoms)
            or branch | core | set(ketone.atoms) != frozenset(range(mol.GetNumAtoms()))):
        return None
    candidates = _spiro_parent_sequence_candidates(mol, core, _ring_system_graph(mol, Chem, core))
    names = []
    for parent, sequence in candidates:
        locants = {idx: locant for locant, idx in enumerate(sequence, 1)}
        key = (locants[nitrogen], locants[ketone.center])
        hydride = attach_suffix(parent, "one", (key[1],))
        name = join_prefix_parent(format_prefixes([(str(key[0]), text)]), hydride)
        names.append((key, name, tuple((idx, str(locant)) for idx, locant in locants.items())))
    if not names:
        return None
    _, name, locants = min(names)
    rule = "P-24/P-64.2"
    return MethodDecision(name, "substitutive", frozenset({ketone.center}),
                          frozenset(range(mol.GetNumAtoms())), rule, locants,
                          preference_issues=(SelectionLimitation(
                              "spiro-ketone-parent", rule, "The mapped ketone and sulfonyl group are cited."),))


@lru_cache(maxsize=2)
def _sultam_spiro_query(Chem):
    return Chem.MolFromSmarts(
        "[S;X4]1(=[O;D1])(=[O;D1])-[N;X3]-[C;X4]2(-[C]-[N;X3]-[C]-[C]-2)"
        "-[c]2[c][c][c][c][c]-12"
    )


def sultam_spiro_dione(mol, Chem):
    """Cite carbonyl and sulfur dioxide suffixes on their mapped components."""
    from .component_mapping import MappedComponent, decorate_component
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation

    if mol.GetRingInfo().NumRings() != 3 or mol.GetStereoGroups():
        return None
    matches = mol.GetSubstructMatches(_sultam_spiro_query(Chem), maxMatches=3)
    if not matches or len(matches) == 3 or any(
            a.GetFormalCharge() or a.GetIsotope() or a.GetChiralTag()
            or a.GetNumRadicalElectrons() or a.HasQuery() for a in iter_atoms(mol)):
        return None
    required = frozenset(range(mol.GetNumAtoms()))
    ring_atoms = frozenset(i for ring in mol.GetRingInfo().AtomRings() for i in ring)
    rows = []
    for match in matches:
        s, o1, o2, n, junction, c2, imide_n, c5, c4, *benzene = match
        core = frozenset(match) - {o1, o2}
        if ring_atoms != core:
            continue
        carbonyls = []
        for c in (c2, c5):
            oxygen = [b.GetOtherAtomIdx(c) for b in mol.GetAtomWithIdx(c).GetBonds()
                      if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtomIdx(c) not in core
                      and b.GetOtherAtomIdx(c) not in {o1, o2}
                      and b.GetOtherAtom(mol.GetAtomWithIdx(c)).GetAtomicNum() == 8
                      and b.GetOtherAtom(mol.GetAtomWithIdx(c)).GetDegree() == 1]
            if len(oxygen) != 1:
                break
            carbonyls.extend(oxygen)
        if len(carbonyls) != 2:
            continue
        suffix_oxygens = {o1, o2, *carbonyls}
        if any(bond.GetBondTypeAsDouble() > 1
               and (bond.GetBeginAtomIdx() in core) != (bond.GetEndAtomIdx() in core)
               and not ({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & suffix_oxygens)
               for bond in iter_bonds(mol)):
            continue
        mapping = dict(zip((s, n, junction, *benzene, imide_n, c2, c4, c5),
                           ("1", "2", "3", "3a", "4", "5", "6", "7", "7a",
                            "1'", "2'", "4'", "5'"), strict=True))
        names = _mapped_spiro_branches(mol, Chem, core, required, None)
        if names is None or any(host in {s, junction, c2, c5} for host, _ in names):
            continue
        parent = "2H-spiro[1,2-benzothiazole-3,3'-pyrrolidine]-2',5'-dione 1,1-dioxide"
        prefixes = tuple((mapping[i], text) for (i, _), text in names.items())
        component = MappedComponent(parent, tuple(mapping.items()), core, prefixes,
                                    required - core, systematic_parent=True)
        name = decorate_component(mol, Chem, component)
        if name:
            rows.append((name, tuple(mapping.items()), frozenset({c2, c5})))
    if len({row[0] for row in rows}) != 1:
        return None
    name, mapping, centers = rows[0]
    rule = "P-24/P-64.2/P-62.5"
    return MethodDecision(name, "substitutive", centers, frozenset(range(mol.GetNumAtoms())),
                          rule, mapping, preference_issues=(SelectionLimitation(
                              "spiro-parent", rule, "The imide and sulfur oxide groups occupy distinct mapped sites."),))
