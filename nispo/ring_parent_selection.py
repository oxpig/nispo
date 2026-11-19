"""Graph and renderer evidence for P-44 aromatic and polycyclic parents.

This module contains no molecule/name preference table. Ring type is proved
from the induced skeleton; unimplemented type-specific criteria stay unknown.
Only a renderer can declare the hydride, its numbering and its prefix units.
"""

from collections import Counter
from .parent_selection import replace_parent_facts as replace
import re

from .molecule_iteration import iter_bonds
from .molecule_cache import mol_cache_dict
from .graph_model import graph_view
from .parent_selection import (RING_TYPE_ORDER, graph_parent_facts, locant_set,
                               principal_context)


def aromatic_bond_counts(mol, atoms):
    """Count bonds, never use a Kekule placement as aromatic numbering.

    The caller has proved a closed parent skeleton: an aromatic component may
    not be split at a fusion/bridge. Its total double count is Kekule invariant.
    """
    from rdkit import Chem

    cache = mol_cache_dict(mol, "parent_kekule_counts")
    key = frozenset(atoms)
    if key not in cache:
        views = mol_cache_dict(mol, "parent_kekule_view")
        if "mol" not in views:
            view = Chem.Mol(mol)
            try:
                Chem.Kekulize(view, clearAromaticFlags=True)
            except (ValueError, RuntimeError):
                view = None
            views["mol"] = view
        view = views["mol"]
        if view is None:
            cache[key] = (None, None)
        else:
            orders = [b.GetBondTypeAsDouble() for b in view.GetBonds()
                      if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= key]
            cache[key] = (sum(v in {2, 3} for v in orders), orders.count(2))
    return cache[key]


def _skeleton(mol, atoms):
    """Induced graph with no external substitution or hydrogen annotations."""
    from rdkit import Chem

    cache = mol_cache_dict(mol, "parent_ring_skeleton")
    key = frozenset(atoms)
    if key not in cache:
        graph, mapping = Chem.RWMol(), {}
        for idx in sorted(key):
            source = mol.GetAtomWithIdx(idx)
            atom = Chem.Atom(source.GetAtomicNum())
            atom.SetIsAromatic(source.GetIsAromatic())
            atom.SetNoImplicit(True)
            mapping[idx] = graph.AddAtom(atom)
        for bond in iter_bonds(mol):
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in key and b in key:
                graph.AddBond(mapping[a], mapping[b], bond.GetBondType())
        graph = graph.GetMol()
        graph.UpdatePropertyCache(strict=False)
        rings = tuple(tuple(ring) for ring in Chem.GetSymmSSSR(graph))
        topology = Chem.Mol(graph)
        for atom in topology.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in topology.GetBonds():
            bond.SetIsAromatic(False)
            bond.SetBondType(Chem.BondType.SINGLE)
        topology.UpdatePropertyCache(strict=False)
        identity = Chem.MolToSmiles(topology, canonical=True, allBondsExplicit=True)
        inverse = {value: idx for idx, value in mapping.items()}
        cache[key] = (identity, tuple(tuple(inverse[i] for i in ring) for ring in rings))
    return cache[key]


def _blocks(graph):
    """Biconnected edge blocks; articulation separates spiro components."""
    discovery, low, stack, result = {}, {}, [], []

    def visit(a, parent=None):
        discovery[a] = low[a] = len(discovery)
        for b in sorted(graph[a]):
            if b == parent:
                continue
            if b not in discovery:
                stack.append((a, b))
                visit(b, a)
                low[a] = min(low[a], low[b])
                if low[b] >= discovery[a]:
                    block = []
                    while stack:
                        edge = stack.pop()
                        block.append(edge)
                        if edge == (a, b):
                            break
                    result.append(block)
            elif discovery[b] < discovery[a]:
                stack.append((a, b))
                low[a] = min(low[a], discovery[b])
    visit(next(iter(graph)))
    return result


def _ring_topology(mol, atoms):
    """Share unnumbered topology across renderings of one physical parent.

    Only primitive original-graph facts are retained. Parent vocabulary,
    locants, hydrogen state and component-construction proofs stay live.
    """
    cache = mol_cache_dict(mol, "parent_ring_topology")
    if atoms not in cache:
        view = graph_view(mol, None)
        graph = {i: frozenset(n for n in view.neighbors[i] if n in atoms)
                 for i in atoms}
        rank = sum(map(len, graph.values())) // 2 - len(atoms) + 1
        blocks = tuple(tuple(edges) for edges in _blocks(graph)) if rank > 1 else ()
        cache[atoms] = graph, rank, blocks
    return cache[atoms]


def _horizontal_row(rings):
    """Maximum straight row for a proved, hole-free six-membered face system.

    In an internal hexagon a straight row crosses opposite fusion edges.
    This also covers peri-fused polyhexes if the supplied facial rings give
    one simple external boundary. Other ring shapes and ambiguous face sets
    need an explicit geometric producer.
    """
    if len(rings) == 2:
        left, right = rings
        common = set(left) & set(right)
        if len(common) != 2:
            return None
        edge = frozenset(common)
        return (2 if all(edge in {frozenset((ring[i], ring[(i + 1) % len(ring)]))
                                  for i in range(len(ring))} for ring in rings) else None)
    if len(rings) < 2 or any(len(ring) != 6 or len(set(ring)) != 6 for ring in rings):
        return None
    sets = list(map(set, rings))
    ring_edges = [tuple(frozenset((ring[i], ring[(i + 1) % 6])) for i in range(6))
                  for ring in rings]
    edge_counts = Counter(edge for edges in ring_edges for edge in edges)
    if any(count > 2 for count in edge_counts.values()):
        return None
    adjacent = {i: set() for i in range(len(rings))}
    for i in range(len(rings)):
        for j in range(i):
            common = sets[i] & sets[j]
            if len(common) > 2:
                return None
            if len(common) == 2:
                edge = frozenset(common)
                if edge not in ring_edges[i] or edge not in ring_edges[j]:
                    return None
                adjacent[i].add(j)
                adjacent[j].add(i)
    # An edge-fused polyhex with holes has multiple boundary cycles; a basis
    # of six-cycles alone does not establish which drawing is the source face
    # set. Decline it rather than assigning an accidental row.
    boundary = [tuple(edge) for edge, count in edge_counts.items() if count == 1]
    boundary_graph = {}
    for a, b in boundary:
        boundary_graph.setdefault(a, set()).add(b)
        boundary_graph.setdefault(b, set()).add(a)
    if not boundary_graph or any(len(v) != 2 for v in boundary_graph.values()):
        return None
    visited, stack = set(), [next(iter(boundary_graph))]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(boundary_graph[node] - visited)
    if visited != set(boundary_graph):
        return None
    face_seen, stack = set(), [0]
    while stack:
        node = stack.pop()
        if node not in face_seen:
            face_seen.add(node)
            stack.extend(adjacent[node] - face_seen)
    if len(face_seen) != len(rings):
        return None

    def opposite(previous, current, following):
        edges = ring_edges[current]
        incoming = edges.index(frozenset(sets[previous] & sets[current]))
        outgoing = edges.index(frozenset(sets[following] & sets[current]))
        return (incoming - outgoing) % 6 == 3

    def extend(path):
        return max([len(path)] + [extend(path + [n]) for n in adjacent[path[-1]]
                   if n not in path and (len(path) < 2 or opposite(path[-2], path[-1], n))])
    return max(extend([i]) for i in adjacent)


def ring_type_evidence(mol, atoms, locants, parent_name, *, include_components=True):
    """Return (type, sequential detail facts, fixed-parent identity)."""
    atoms = frozenset(atoms)
    graph, rank, blocks = _ring_topology(mol, atoms)
    identity, rings = _skeleton(mol, atoms)
    # Hydrogenation and indicated-H operations do not change the intrinsic
    # fusion/bridge construction. They have their own later P-44.4 criteria.
    fixed_parent = re.sub(r"(?:\d+[a-z]*\'*H[,-]?)+", "", parent_name)
    fixed_parent = re.sub(r"(?:(?:\d+[a-z]*\'*,)*\d+[a-z]*\'*-)?(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?(?:de)?hydro", "", fixed_parent)
    identity = (fixed_parent.strip('-'), identity)
    if 'hydro' in parent_name and hydride_numbering(parent_name)[1] is None:
        identity = None
    if rank <= 1:
        return None, None, identity
    if any(len(edges) < 3 for edges in blocks):
        return None, None, identity  # Acyclic links require phane/assembly rules.
    if len(blocks) > 1:
        counts = Counter(i for edges in blocks for i in {j for edge in edges for j in edge})
        spiro = [i for i, count in counts.items() for _ in range(count - 1)]
        monocyclic = all(len(edges) == len({i for edge in edges for i in edge}) for edges in blocks)
        saturated = monocyclic and all(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
                                      for edges in blocks for a, b in edges)
        # For saturated monocyclic components the spiro locants complete the
        # type-specific comparison. Other component seniorities remain unknown.
        details = (-len(spiro), -int(saturated),
                   locant_set(locants[i] for i in spiro) if saturated else (),
                   () if saturated else None)
        return RING_TYPE_ORDER["spiro"], details, identity
    ring_sets = list(map(set, rings))
    fused = len(rings) == rank and all(
        len(a & b) <= 1 or len(a & b) == 2 and mol.GetBondBetweenAtoms(*sorted(a & b)) is not None
        for j, a in enumerate(ring_sets) for b in ring_sets[j + 1:])
    if fused:
        sizes = tuple(-n for n in sorted(map(len, rings), reverse=True))
        row = _horizontal_row(rings)
        # Fully explicit, simple fusion descriptors prove the next two
        # criteria. Missing descriptors/numbers stay unknown; a retained name
        # does not imply an empty descriptor. Component seniority needs P-25.
        descriptors = re.findall(r"\[(?:(\d+(?:,\d+)*)-)?([a-z])\]", parent_name)
        letters = numbers = None
        if len(descriptors) == rank - 1:
            letters = tuple(sorted(letter for _, letter in descriptors))
            if all(text for text, _ in descriptors):
                numbers = tuple(int(value) for text, _ in descriptors for value in text.split(','))
        from .fusion_components import (fusion_construction, component_list_key, fusion_input_scope,
                                        fusion_descriptor_evidence)
        from .fusion_networks import matches_skeleton
        construction = (fusion_construction(parent_name)
                        if include_components and fusion_input_scope(mol) else None)
        if construction is not None and not matches_skeleton(construction,mol,atoms):
            construction = None
        component_key = component_list_key(construction)
        if construction is not None:
            # Internal rings of a retained component are not additional
            # component-fusion descriptors. P-25.3.8 also proves the omitted
            # 1,2 numbers of a terminal monocyclic hydrocarbon component.
            letters,numbers = fusion_descriptor_evidence(construction)
        return RING_TYPE_ORDER["fused"], (sizes, -row if row is not None else None,
                                         letters, numbers, component_key), identity
    # A theta graph proves a nonfused bicyclic system; the renderer supplies
    # its three bridge lengths. Do not infer higher von Baeyer descriptors.
    descriptor = re.search(r"bicyclo\[(\d+)\.(\d+)\.(\d+)\]", parent_name)
    if rank == 2 and sorted(map(len, graph.values())).count(3) == 2 and all(len(v) in {2, 3} for v in graph.values()):
        sizes = tuple(map(int, descriptor.groups())) if descriptor else None
        heads = [i for i in graph if len(graph[i]) == 3]
        paths = []
        for first in graph[heads[0]]:
            previous, current, length = heads[0], first, 0
            while current != heads[1]:
                length += 1
                previous, current = current, next(n for n in graph[current] if n != previous)
            paths.append(length)
        if sizes and sizes[-1] > 0 and sizes == tuple(sorted(paths, reverse=True)):
            return RING_TYPE_ORDER["nonfused-bridged"], (sizes, (), ()), identity
    return None, None, identity


def hydride_numbering(parent_name):
    """Read explicit hydrogenation operations from the renderer's core name.

    No Kekule bond locant is ever read here. Unrecognized operation syntax is
    a barrier, rather than an empty set of hydrogenation locants.
    """
    indicated = re.findall(r"(\d+[a-z]*\'*)H", parent_name)
    hydro = re.findall(r"(?:^|-)([\d,a-z']+)-(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?(?:de)?hydro", parent_name)
    if "hydro" in parent_name and not hydro:
        return indicated, None, None
    operations = re.findall(r"(?:^|-)([\d,()]+)-(?:di|tri|tetra|penta|hexa|hepta|octa)?(en|yn)(?:e|$)", parent_name)
    all_locs = [loc for text in hydro for loc in text.split(',')]
    double_locs = []
    for text, ending in operations:
        all_locs.extend(text.split(','))
        if ending == "en":
            double_locs.extend(text.split(','))
    if (hydro and operations) or ('dehydro' in parent_name and re.search(r'(?<!de)hydro', parent_name)):
        return indicated, None, None  # Mixed additive/subtractive operations need P-31 detail.
    # Unlocanted unsaturated systematic hydrides need explicit numbering.
    if not operations and re.search(r"(?:cyclo|spiro)\[.*\].*(?:en|yn)e?$", parent_name):
        return indicated, None, None
    return indicated, all_locs, double_locs


def rendered_ring_facts(mol, atoms, locants, parent_name, prefixes, name, *,
                        suffix_centers=(), suffix_locants=(), replacement=False,
                        attachment_locants=()):
    """Evidence for an explicit, ordinary neutral ring-parent rendering."""
    context = principal_context(mol)
    if context is None:
        return None
    rank, groups = context
    if groups and not suffix_centers:
        return None
    if not set(suffix_centers) <= {g.center for g in groups}:
        return None
    atoms, locants = frozenset(atoms), dict(locants)
    if not atoms or not atoms <= locants.keys():
        return None
    view = graph_view(mol, None)
    if not all(view.ring_atoms[i] for i in atoms):
        return None
    indicated, unsaturation, double = hydride_numbering(parent_name)
    # A neutral aromatic donor N of degree two retains the hydride's indicated
    # hydrogen position when an external substituent replaces its hydrogen.
    # Bridgehead N atoms of degree three are not indicated-H sites.
    indicated = set(indicated) | {locants[i] for i in atoms
        if view.atomic_nums[i] == 7
        and view.aromatic_atoms[i]
        and sum(n in atoms for n in view.neighbors[i]) == 2
        and (mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 or
             any(n not in atoms for n in view.neighbors[i]))}
    if (not indicated and 'hydro' not in parent_name
            and any(view.aromatic_atoms[i] for i in atoms)
            and any(not view.aromatic_atoms[i] for i in atoms)):
        indicated = None  # A retained partially mancude hydride may have implicit C-H indication.
    facts = graph_parent_facts(mol, atoms, locants, principal_class=rank,
        suffix_centers=suffix_centers, suffix_locants=suffix_locants, prefixes=prefixes,
        kind="ring", name=name, replacement=replacement, indicated_h=indicated,
        attachment_locants=attachment_locants)
    if facts is None:
        return None
    if (not unsaturation and re.fullmatch(r'cyclo[a-z]+(?:en|ene|yn|yne)', parent_name)
            and facts.multiple_count):
        # The sole double/triple bond of a simple monocycle has implicit
        # locant 1. Do not treat an omitted locant as an empty locant set.
        unsaturation = ['1'] if facts.multiple_count == 1 else None
        double = ['1'] if facts.multiple_count == facts.double_count == 1 else (
            [] if facts.multiple_count == 1 else None)
    ring_type, details, identity = ring_type_evidence(mol, atoms, locants, parent_name)
    from .fusion_components import fusion_construction, fusion_input_scope
    from .fusion_networks import matches_skeleton
    construction = (fusion_construction(parent_name)
                    if ring_type == RING_TYPE_ORDER['fused'] and fusion_input_scope(mol) else None)
    if construction is not None and not matches_skeleton(construction,mol,atoms):
        construction = None
    # Fusion sites are graph junctions only in a proved fused ring system.
    fusion = locant_set(locants[i] for i in atoms if view.atomic_nums[i] == 6
                       and sum(n in atoms for n in view.neighbors[i]) > 2)
    return replace(facts, ring_type=ring_type, ring_type_details=details, ring_identity=identity,
                   fusion_construction=construction, fusion_scope=atoms if construction is not None else None,
                   fusion_locants=fusion if ring_type == RING_TYPE_ORDER["fused"] else (),
                   unsaturation_locants=locant_set(unsaturation) if unsaturation is not None else None,
                   double_locants=locant_set(double) if double is not None else None)


def record_rendered_ring_facts(mol, atoms, locants, parent_name, prefixes, name, **kwargs):
    facts = rendered_ring_facts(mol, atoms, locants, parent_name, prefixes, name, **kwargs)
    mol_cache_dict(mol, "rendered_parent_selection")[(name, frozenset(atoms))] = facts
    return facts


def recorded_ring_facts(mol, name, atoms):
    return mol_cache_dict(mol, "rendered_parent_selection").get((name, frozenset(atoms)))


def component_spiro_parent_facts(mol, atoms, locants, parent_name, name):
    """General seniority and fusion count for discrete-component spiro names.

    Component ordering, component construction and primed spiro numbering need
    their own P-24/P-44 evidence. Generic spiro locants must not be compared to
    an incomplete discrete-component construction.
    """
    facts = rendered_ring_facts(mol, atoms, locants, parent_name, None, name)
    if facts is None or facts.ring_type != RING_TYPE_ORDER['spiro']:
        return None
    return replace(facts, ring_type_details=(facts.ring_type_details[0], None, None, None),
                   ring_identity=None)


def template_parent_frontier(mol, records):
    """Apply proved structural criteria before a template render budget.

    A topology match has not yet rendered its hydrogen state or prefixes.
    Neither an implicit descriptor tie nor indicated-H numbering is assumed.
    """
    from .parent_selection import parent_frontier
    from .template_access import template_record_locants, template_record_name

    rows = []
    for record in records:
        mapping = {idx: template_record_locants(record.template).get(j, "")
                   for j, idx in enumerate(record.match)}
        facts = rendered_ring_facts(mol, record.match, mapping,
            template_record_name(record.template), None, template_record_name(record.template))
        if facts is not None:
            facts = replace(facts, indicated_h_locants=None, ring_identity=None)
        rows.append((record, facts))
    return tuple(record for record, _ in parent_frontier(rows, lambda row: row[1]))
