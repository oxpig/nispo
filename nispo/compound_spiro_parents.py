"""Mapped P-24.5 parents with a monocyclic and a polycyclic unit.

The two components supply proved complete hydrides and fixed numberings.
The shared ring renderer then owns the whole spiro system, including rooted
prefixes, characteristic suffixes and original-graph stereo decoration.
"""
from itertools import product
from dataclasses import dataclass

from .constants import MULT
from .formatting import prefix_alphabetization_key
from .parent_selection import locant_key


@dataclass(frozen=True, slots=True)
class _SpiroHydrideState:
    reference_hydrogens: tuple[int, ...]


def skeletons(graph, Chem):
    yield from _skeletons(graph, Chem, allow_unsaturated=False)


def unsaturated_skeletons(graph, Chem):
    yield from _skeletons(graph, Chem, allow_unsaturated=True)


def _skeletons(graph, Chem, *, allow_unsaturated):
    from .spiro_component_rules import (
        _two_component_splits_by_articulation, _cycle_rank_for_atoms,
        _submol_for_atoms, _proved_hydride_component_names, _prime_locant,
    )

    numbers = tuple(a.GetAtomicNum() for a in graph.GetAtoms())
    rank = graph.GetNumBonds() - graph.GetNumAtoms() + 1
    unsaturated = any(b.GetBondTypeAsDouble() != 1 for b in graph.GetBonds())
    if (graph.GetNumAtoms() > 24 or rank not in ({3, 4} if allow_unsaturated else {3})
            or any(n not in {6, 7, 8, 16} for n in numbers)
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetTotalValence() != {6:4, 7:3, 8:2, 16:2}[a.GetAtomicNum()]
                   for a in graph.GetAtoms())
            or unsaturated != allow_unsaturated):
        return
    adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in graph.GetAtoms()}
    vertices = set(adjacency)
    splits = _two_component_splits_by_articulation(adjacency, vertices)
    if len(splits) != 1:
        return
    junction, sides = next(iter(splits.items()))
    if numbers[junction] != 6 or len(sides) != 2 or len(adjacency[junction]) != 4:
        return
    units = []
    for side in sides:
        atoms = set(side) | {junction}
        rank = _cycle_rank_for_atoms(adjacency, atoms)
        if rank not in {1, 2, 3}:
            return
        prepared = _submol_for_atoms(graph, Chem, atoms)
        if prepared is None:
            return
        submol, to_original, to_sub = prepared
        components = _proved_hydride_component_names(submol, Chem, to_sub[junction])
        if rank > 1 and not allow_unsaturated:
            components = [c for c in components if c.parent_name.startswith('bicyclo[')]
        if allow_unsaturated:
            components = [c for c in components if c.junction_proved]
        if not components:
            return
        units.append((rank, components, to_original))
    if sorted(unit[0] for unit in units) not in ([1, 2], [1, 3]):
        return
    rows = []
    for left, right in product(units[0][1], units[1][1]):
        ordered = sorted(((left, units[0][2]), (right, units[1][2])),
                         key=lambda item: prefix_alphabetization_key(item[0].parent_name))
        first, first_map = ordered[0]
        second, second_map = ordered[1]
        # P-24.5 fixed component numbering precedes decorations. Compare
        # spiro locants only between these same intrinsic component names.
        fixed = ((first.parent_name, second.parent_name),
                 locant_key(first.spiro_locant), locant_key(second.spiro_locant))
        mapping = {first_map[i]: loc for i, loc in first.locants.items()}
        for i, loc in second.locants.items():
            mapping.setdefault(second_map[i], _prime_locant(loc))
        if set(mapping) != vertices or len(set(mapping.values())) != len(vertices):
            continue
        replacements = list(first.replacement_prefixes)
        replacements.extend((_prime_locant(loc), term) for loc, term in second.replacement_prefixes)
        # Skeletal replacement citation follows O > S > N, independently
        # of alphabetization of detachable characteristic prefixes.
        parts = []
        for term in ('oxa', 'thia', 'aza'):
            locations = sorted((loc for loc, word in replacements if word == term), key=locant_key)
            if locations:
                count = len(locations)
                if count > 1 and count not in MULT:
                    return
                parts.append(','.join(locations)+'-'+(MULT[count] if count > 1 else '')+term)
        if any(term not in {'oxa', 'thia', 'aza'} for _, term in replacements):
            continue
        parent = ('-'.join(parts)+'-' if parts else '') + (
            f"spiro[{first.parent_name}-{first.spiro_locant},"
            f"{_prime_locant(second.spiro_locant)}-{second.parent_name}]")
        sequence = tuple(sorted(mapping, key=lambda i: locant_key(mapping[i])))
        inverse = {i: j for j, i in enumerate(sequence)}
        edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                      for b in graph.GetBonds())
        row = (parent, sequence, tuple(mapping[i] for i in sequence),
               tuple(numbers[i] for i in sequence), edges, True, fixed, 'component-spiro')
        if allow_unsaturated:
            hydrogens = tuple(graph.GetAtomWithIdx(i).GetTotalNumHs() for i in sequence)
            row += (_SpiroHydrideState(hydrogens),)
        rows.append(row)
    if rows:
        minimum = min(row[6] for row in rows)
        seen = set()
        for row in rows:
            if row[6] == minimum and row[1:3] not in seen:
                seen.add(row[1:3])
                yield row
