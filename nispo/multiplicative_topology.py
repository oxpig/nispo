"""Mapped topology evidence for bounded multiplicative-parent selection.

The module checks structural identity, disjoint unit ownership, attachment
edges, and the complete connected complement of a full-center unit pack. It
does not render names or choose a nomenclature method. Ambiguous decompositions
fail closed.
"""

from collections import deque
from dataclasses import dataclass

from .molecule_iteration import iter_bonds


@dataclass(frozen=True, slots=True)
class RepeatedUnit:
    """One mapped candidate unit from the parent-unit enumerator.

    The field names align with ``naming_methods._Unit``. Original atom IDs are
    retained throughout; ``root`` is the attachment atom inside the unit and
    ``connector`` is the atom outside it.
    """

    atoms: frozenset
    root: object
    connector: object
    signature: str
    parent: str
    locant: str
    centers: frozenset = frozenset()
    rank: tuple = ()


@dataclass(frozen=True, slots=True)
class MultiplicativeTopology:
    """Evidence for one unambiguous maximal repeated-unit decomposition."""

    status: str
    identity_key: tuple
    maximal_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]
    unit_atom_sets: tuple[tuple[int, frozenset], ...]
    original_atom_ownership: tuple[tuple[int, tuple], ...]
    unit_locants: tuple[tuple[int, str], ...]
    connector_atoms: frozenset
    central_group_atoms: frozenset
    attachment_distances: tuple[tuple[int, int], ...]
    rule: str
    reason: str


def _candidate_key(unit):
    # The rooted graph signature is structural identity. Display parent text
    # is retained as output evidence but cannot make two units chemically
    # identical or different.
    return (unit.signature, str(unit.locant), len(unit.centers))


def _normalize_units(units):
    normalized = []
    for index, unit in enumerate(units):
        atoms = frozenset(unit.atoms)
        if (not atoms or unit.root not in atoms or unit.connector in atoms
                or not unit.signature or not unit.parent or unit.locant is None):
            continue
        normalized.append((index, RepeatedUnit(
            atoms, unit.root, unit.connector, str(unit.signature),
            str(unit.parent), str(unit.locant), frozenset(unit.centers),
            tuple(getattr(unit, "rank", ())),
        )))
    return normalized


def _maximum_disjoint_packings(rows, state_limit):
    """Return all maximum disjoint sets, or None if the bounded search expires."""
    # Repeated discovery routes can yield the exact same candidate. Collapse
    # only exact graph/attachment duplicates, preserving the first input ID.
    deduplicated = {}
    for index, unit in rows:
        key = (unit.atoms, unit.root, unit.connector, unit.locant)
        deduplicated.setdefault(key, (index, unit))
    rows = sorted(deduplicated.values(), key=lambda row: (
        tuple(sorted(row[1].atoms)), row[1].locant, row[1].root, row[0]
    ))
    if len(rows) > 40:
        return None
    best_size = 0
    best = set()
    states = 0

    def search(position, chosen, occupied):
        nonlocal best_size, states
        states += 1
        if states > state_limit:
            raise OverflowError
        if len(chosen) + len(rows) - position < best_size:
            return
        if position == len(rows):
            packed = tuple(chosen)
            if len(packed) > best_size:
                best_size = len(packed)
                best.clear()
            if len(packed) == best_size:
                best.add(tuple(sorted(packed)))
            return
        index, unit = rows[position]
        if not (unit.atoms & occupied):
            search(position + 1, chosen + [index], occupied | unit.atoms)
        search(position + 1, chosen, occupied)

    try:
        search(0, [], frozenset())
    except OverflowError:
        return None
    return tuple(sorted(best))


def _connected_and_tree(adjacency, nodes):
    nodes = set(nodes)
    if not nodes:
        return False
    edges = set()
    for atom in nodes:
        for neighbor in adjacency.get(atom, ()):
            if neighbor in nodes:
                edges.add(frozenset((atom, neighbor)))
    seen = {next(iter(nodes))}
    stack = list(seen)
    while stack:
        atom = stack.pop()
        for neighbor in adjacency.get(atom, ()):
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == nodes and len(edges) == len(nodes) - 1


def _distances(adjacency, start):
    distances = {start: 0}
    queue = deque((start,))
    while queue:
        atom = queue.popleft()
        for neighbor in adjacency.get(atom, ()):
            if neighbor not in distances:
                distances[neighbor] = distances[atom] + 1
                queue.append(neighbor)
    return distances


def _build_mol_adjacency(mol, all_atoms):
    nodes = frozenset(all_atoms)
    adjacency = {atom: set() for atom in nodes}
    orders = {}
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in nodes or right not in nodes:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        orders[frozenset((left, right))] = bond.GetBondTypeAsDouble()
    return nodes, adjacency, orders


def multiplicative_topology(mol, candidates, all_atoms, required_centers=(), *,
                            allow_residual=False, state_limit=100_000):
    """Analyze mapped candidate units under P-51.3 and return explicit evidence.

    ``candidates`` can be the existing ``_Unit`` records. The largest
    unambiguous disjoint set of identical units is identified without using
    name length or rendered spelling. When residual centres are admitted,
    the caller must separately prove and render the connector and its
    pendant substituents. Ambiguous packings remain unproven.
    """
    nodes, adjacency, bond_orders = _build_mol_adjacency(mol, all_atoms)
    return topology_from_graph(nodes, adjacency, candidates,
                               required_centers=required_centers,
                               bond_orders=bond_orders,
                               allow_residual=allow_residual,
                               state_limit=state_limit)


def topology_from_graph(nodes, adjacency, candidates, required_centers=(), *,
                        bond_orders=None, allow_residual=False, state_limit=100_000):
    """Graph-level implementation, also usable with independently built graphs."""
    nodes = frozenset(nodes)
    required_centers = frozenset(required_centers)
    bond_orders = bond_orders or {}
    normalized = _normalize_units(candidates)
    if not nodes or not normalized or len(normalized) > 256:
        return None
    if any(not unit.atoms <= nodes for _, unit in normalized):
        return None

    classes = {}
    for index, unit in normalized:
        classes.setdefault(_candidate_key(unit), []).append((index, unit))
    maximal = []
    for identity, rows in classes.items():
        packings = _maximum_disjoint_packings(rows, state_limit)
        if packings is None:
            return None
        best_count = max(map(len, packings), default=0)
        if best_count >= 2:
            maximal.append((identity, best_count, packings))
    if not maximal:
        return None
    highest = max(item[1] for item in maximal)
    choices = [item for item in maximal if item[1] == highest]
    if len(choices) != 1:
        return None  # P-44/P-45 parent comparison is not part of this helper.
    identity, _, packings = choices[0]
    if len(packings) != 1:
        return None
    selected_indices = packings[0]
    by_index = dict(normalized)
    units = [by_index[index] for index in selected_indices]
    covered = frozenset().union(*(unit.atoms for unit in units))
    connector_atoms = nodes - covered
    if any(unit.connector not in connector_atoms for unit in units):
        return None

    # Prove each candidate is a single mapped unit with one original attachment
    # edge, and the complete molecule is partitioned into those units plus the
    # connector/residual region.
    for unit in units:
        crossing = [(left, right) for left in unit.atoms
                    for right in adjacency.get(left, ())
                    if right in connector_atoms]
        if len(crossing) != 1 or crossing[0] != (unit.root, unit.connector):
            return None
    centers = frozenset().union(*(unit.centers for unit in units))
    if required_centers and (not centers <= required_centers
                             or not allow_residual and centers != required_centers):
        return None
    complete_full_center_pack = not required_centers or centers == required_centers

    # Contract the parent units to terminal markers. In the extended scope
    # the complement may include pendant rings. The connector writer proves
    # that the paths between the terminals stay outside those rings.
    marker = {index: ("unit", index) for index in selected_indices}
    contracted = {atom: set() for atom in connector_atoms}
    contracted.update({marker[index]: set() for index in selected_indices})
    for atom in connector_atoms:
        for neighbor in adjacency.get(atom, ()):
            if neighbor in connector_atoms:
                contracted[atom].add(neighbor)
    for index, unit in zip(selected_indices, units):
        u_marker = marker[index]
        contracted[u_marker].add(unit.connector)
        contracted[unit.connector].add(u_marker)
    connected = (len(_distances(contracted, next(iter(contracted)))) == len(contracted)
                 if allow_residual else _connected_and_tree(contracted, contracted))
    if not connected:
        return None

    # Preserve the bond-order signature for every parent-to-connector bond.
    attachment_orders = {
        index: bond_orders.get(frozenset((unit.root, unit.connector)), 1.0)
        for index, unit in zip(selected_indices, units)
    }
    if len(set(attachment_orders.values())) != 1:
        return None

    degree = {atom: len(neighbors) for atom, neighbors in contracted.items()
              if atom in connector_atoms}
    branch_nodes = tuple(sorted(atom for atom, value in degree.items() if value >= 3))
    central = branch_nodes[0] if len(branch_nodes) == 1 else None
    distances = ()
    status = "maximal_identical_units"
    residual_indices = ()
    reason = "A unique maximum disjoint repeated-unit set is proved; caller must verify connector naming and P-51.3.1 symmetry."
    rule = "P-51.3.1/P-51.3.2.1"

    if len(units) > 3:
        # A single cyclic central group can carry several direct ports. Strip
        # terminal branches to get its cyclic core; the renderer must still
        # prove that it is one numbered ring and owns the entire complement.
        core = set(connector_atoms)
        cyclic_degrees = {i: len(core.intersection(adjacency.get(i, ()))) for i in core}
        leaves = [i for i, count in cyclic_degrees.items() if count < 2]
        while leaves:
            i = leaves.pop()
            core.remove(i)
            for j in core.intersection(adjacency.get(i, ())):
                cyclic_degrees[j] -= 1
                if cyclic_degrees[j] == 1:
                    leaves.append(j)
        cyclic_central = bool(core and {unit.connector for unit in units} <= core
                              and all(cyclic_degrees[i] == 2 for i in core))
        if complete_full_center_pack:
            if central is not None and all(unit.connector == central for unit in units):
                status = "single_multiplicative_group"
                reason = "Every full-center repeated unit attaches directly to one central connector atom."
            else:
                status = "maximal_identical_units"
                reason = ("The complete full-center pack is retained with its whole connected "
                          "complement as central group; nearest-subset evidence is unnecessary.")
        elif cyclic_central and allow_residual:
            status = "single_multiplicative_group"
            reason = "Every repeated unit attaches directly to one cyclic connector."
        elif central is None:
            status = "unproven"
            reason = "More than three occurrences lack one atom-valued central connector group."
        else:
            graph_distances = _distances(contracted, central)
            distances = tuple((index, graph_distances.get(marker[index], -1))
                              for index in selected_indices)
            direct = tuple(index for index in selected_indices
                           if by_index[index].connector == central)
            if len(direct) == len(units):
                status = "single_multiplicative_group"
                reason = "Every identical unit is attached directly to the one central connector atom."
            elif 2 <= len(direct) < len(units):
                if not allow_residual and required_centers and frozenset().union(
                        *(by_index[index].centers for index in direct)) != required_centers:
                    status = "unproven"
                    reason = "The nearest subset would leave principal centers outside the multiplied parent units."
                else:
                    status = "nearest_subset"
                    residual_indices = tuple(index for index in selected_indices if index not in direct)
                    selected_indices = direct
                    units = [by_index[index] for index in selected_indices]
                    reason = ("The maximal set has more than three identical units; the unique central atom has at least two direct unit attachments, "
                              "while remaining identical units are farther away, as required by P-51.3.2.2.")
                    rule = "P-51.3.2.2"
            else:
                status = "unproven"
                reason = "Fewer than two identical units are nearest to the unique central connector atom."
    elif central is not None and all(unit.connector == central for unit in units):
        status = "single_multiplicative_group"
        reason = "All repeated units attach directly to the same central connector atom."

    maximal_indices = packings[0]
    ownership = tuple((index, tuple(sorted(by_index[index].atoms)))
                      for index in maximal_indices)
    locants = tuple((index, str(by_index[index].locant))
                    for index in maximal_indices)
    unit_sets = tuple((index, by_index[index].atoms) for index in maximal_indices)
    final_selected = tuple(selected_indices) if status != "unproven" else ()
    final_residual = tuple(residual_indices) if status == "nearest_subset" else ()
    central_group = (frozenset(connector_atoms) if complete_full_center_pack
                     and status != "nearest_subset" else
                     frozenset(core) if len(units) > 3 and cyclic_central else
                     frozenset((central,)) if central is not None else frozenset())
    return MultiplicativeTopology(
        status=status, identity_key=identity, maximal_indices=maximal_indices,
        selected_indices=final_selected, residual_indices=final_residual,
        unit_atom_sets=unit_sets,
        original_atom_ownership=ownership, unit_locants=locants,
        connector_atoms=frozenset(connector_atoms),
        central_group_atoms=central_group,
        attachment_distances=distances, rule=rule, reason=reason,
    )
