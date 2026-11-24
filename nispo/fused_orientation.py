"""Source-constrained planar embeddings for small fused ring face trees.

The Blue Book fixes preferred ring drawings before fused-system numbering.
This module starts with the illustrated P-25.3.2.3.1 shapes for three- through
eight-membered rings and, where necessary, considers edge-compatible
distortions of a smallest ring. It scores orientations by P-25.3.2.3.3 and
returns clockwise perimeter seeds for the graph-only numbering frontier. It
does not infer a drawing from atom indices, SMILES order, or a name string.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
import math

from nispo.catacondensed_numbering import (
    PeripheralOrientationSeed,
    peripheral_numbering_frontier,
)


_EPS = 1e-7
# Engineering bounds for this rule application; neither is a P-25 limit.
_MAX_FACES = 6
_MAX_FACE_TREE_DEGREE = 3
_MAX_EMBEDDINGS = 50_000
_MAX_EMBEDDING_WORK = 200_000
_ROOT_SOURCE = "Blue Book P-25.3.2.3.1 / P-25.3.2.3.3"


@dataclass(frozen=True, slots=True)
class _Shape:
    size: int
    variant: str
    vertices: tuple[tuple[float, float], ...]

    @property
    def row_axis_center(self):
        # In these templates the horizontal bisector is y=0. Its intersection
        # with the vertical symmetry axis is the mean x coordinate; this is
        # deliberately distinct from the polygon area centroid.
        if self.variant.startswith('pentagon_side_'):
            # FR-5.1.2 removes a different vertex of the same hexagonal
            # grid cell. Its grid centre does not move toward the remaining
            # five vertices when the elongated edge changes direction.
            return (0.0, 0.0)
        return (sum(x for x, _ in self.vertices) / self.size, 0.0)


@dataclass(frozen=True, slots=True)
class GeometricEmbedding:
    atom_coordinates: tuple[tuple[int, float, float], ...]
    face_shapes: tuple[tuple[tuple[int, ...], str], ...]
    face_centers: tuple[tuple[tuple[int, ...], float, float], ...]
    horizontal_row_faces: tuple[frozenset[int], ...]
    horizontal_row_count: int
    row_bond_separation: float
    upper_right_ring_fraction: float
    lower_left_ring_fraction: float
    above_row_ring_fraction: float
    criterion_key: tuple[float, ...]
    numbering_face_path: tuple[tuple[int, ...], ...]
    numbering_transition_fusion_edges: tuple[tuple[int, int], ...]
    numbering_successor_perimeter_edges: tuple[tuple[int, int], ...]
    clockwise_order: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ClockwiseNumberingPath:
    clockwise_order: tuple[int, ...]
    face_path: tuple[tuple[int, ...], ...]
    transition_fusion_edges: tuple[tuple[int, int], ...]
    successor_perimeter_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FusedOrientationFrontier:
    admitted: bool
    reason: str | None
    graph_frontier: object | None = None
    candidates: tuple[GeometricEmbedding, ...] = ()
    orientation_seeds: tuple[PeripheralOrientationSeed, ...] = ()


class _EmbeddingSearchLimit(Exception):
    """Internal signal: a partial embedding frontier cannot be ranked safely."""


def fused_orientation_frontier(mol, *, core_atoms=None, _traversal_root=None,
                               allow_three_face_n_junction=False,
                               allow_multi_face_n_junction=False,
                               allow_larger_system=False,
                               allow_shared_carbon_junctions=False,
                               allow_peri_fused=False, polynaphthylene_core=False):
    """Enumerate best embeddings for a bounded 2–6-face catacondensed tree.

    Supported face shapes are restricted to finite schematic templates in
    P-25.3.2.3.1 Figures 583–589: 3–8-membered rings, including the two
    illustrated phases for triangles, pentagons, hexagons, heptagons, and the
    vertex- and edge-centered octagon phases. If those shapes do not suffice,
    the larger-system path also considers a single smallest ring distorted by
    independently scaling its coordinate axes to match an adjacent ring edge.
    Other distortions, peri-fused systems, and placements requiring geometry
    beyond these schematic templates remain outside this constructor. The
    admitted face graph may branch to degree three; numbering also requires a
    connected horizontal row and source-supported numbering starts on the
    selected uppermost faces. A recognized polynaphthylene may also use the
    FR-5.1 stretched octagon or decagon and hexagons fused outside the row.
    """
    graph = peripheral_numbering_frontier(
        mol, core_atoms=core_atoms,
        allow_three_face_n_junction=allow_three_face_n_junction,
        allow_multi_face_n_junction=allow_multi_face_n_junction,
        allow_larger_system=allow_larger_system,
        allow_shared_carbon_junctions=allow_shared_carbon_junctions,
        allow_peri_fused=allow_peri_fused)
    if not graph.admitted:
        return FusedOrientationFrontier(False, "graph frontier declined: " + str(graph.reason), graph)
    faces = tuple(frozenset(face) for face in graph.ring_faces)
    if len(faces) < 2 or (not allow_larger_system and len(faces) > _MAX_FACES):
        return FusedOrientationFrontier(False, "face count outside bounded two-to-six-face scope", graph)
    sizes = tuple(map(len, faces))
    if polynaphthylene_core and (sizes.count(6) != len(sizes) - 1 or not set(sizes) <= {6, 8, 10}):
        return FusedOrientationFrontier(False, "requires one octagon or decagon with hexagonal arms", graph)
    if any(not 3 <= size <= 8 and not (polynaphthylene_core and size == 10) for size in sizes):
        return FusedOrientationFrontier(False, "ring size outside illustrated 3–8 templates", graph)
    proved_five_seven_six = (sorted(map(len, faces)) == [5, 6, 7]
                             and _bounded_five_seven_six_oxygen_tree(mol, faces))
    proved_seven_face_tree = (sorted(map(len, faces)) in ([5, 7], [6, 7], [7, 7])
                              or proved_five_seven_six)
    # Keep the existing template path for all other ring trees. The corrected
    # source-drawn 7-ring outline is an additional bounded continuation, not
    # a general permission to renumber previously unproved 7-ring systems.
    # The general component route admits the illustrated seven-membered
    # outline wherever it can establish a complete compatible face drawing.
    shape_provider = (_seven_fused_shapes if proved_seven_face_tree
                      or allow_larger_system and any(len(face) == 7 for face in faces)
                      else _shapes)
    if allow_larger_system:
        ordinary_shapes = shape_provider
        def shape_provider(size):
            return _nonrow_pentagon_shapes() if size == 5 else ordinary_shapes(size)
    if polynaphthylene_core:
        shape_provider = _polynaphthylene_shapes
    if not allow_peri_fused and len(graph.fusion_edges) != len(faces) - 1:
        return FusedOrientationFrontier(False, "face graph is not a tree", graph)

    face_edges = {}
    face_adjacency = {i: set() for i in range(len(faces))}
    for i, j in permutations(range(len(faces)), 2):
        if j <= i:
            continue
        common = faces[i] & faces[j]
        if common:
            if (allow_shared_carbon_junctions or allow_peri_fused) and len(common) == 1:
                continue  # A graph-proved junction may touch at one carbon.
            if len(common) != 2 or frozenset(common) not in graph.fusion_edges:
                return FusedOrientationFrontier(False, "face pair is not joined by one shared edge", graph)
            edge = frozenset(common)
            face_edges[(i, j)] = edge
            face_adjacency[i].add(j)
            face_adjacency[j].add(i)
    if not _connected(face_adjacency):
        return FusedOrientationFrontier(False, "face tree is disconnected", graph)
    if not allow_larger_system and any(len(neighbors) > _MAX_FACE_TREE_DEGREE for neighbors in face_adjacency.values()):
        return FusedOrientationFrontier(False, "face-tree branching exceeds bounded degree-three scope", graph)

    face_cycles = []
    for face in faces:
        cycle = _face_cycle_orders(mol, face)
        if not cycle:
            return FusedOrientationFrontier(False, "a face is not a simple induced cycle", graph)
        face_cycles.append(cycle)

    raw = []
    work = [0]
    # Every finite tree has a leaf. Fixing any leaf as the traversal root is
    # complete. Every rigid embedding assigns that leaf one
    # of the enumerated source shapes and cyclic atom-to-vertex maps. Child
    # placement below uses only a translation: it never rotates a child
    # template. Therefore any complete embedding rooted at another face can
    # be globally translated so the selected leaf has its template origin;
    # the leaf's shape/map occurs in this enumeration, and each remaining
    # face is recovered from its shared-edge vector. This is a bijection up to
    # global translation, which does not affect row or quadrant criteria.
    leaves = tuple(i for i, neighbors in face_adjacency.items()
                   if len(neighbors) <= 1)
    roots = leaves or tuple(range(len(faces)))
    traversal_root = min(roots)
    if _traversal_root is not None:
        if (isinstance(_traversal_root, bool) or
                not isinstance(_traversal_root, int) or
                _traversal_root not in roots):
            return FusedOrientationFrontier(
                False, "private traversal root must identify a face-tree leaf", graph)
        traversal_root = _traversal_root
    def search(face_shapes=None):
        root = traversal_root
        shapes = (face_shapes[root] if face_shapes and root in face_shapes
                  else shape_provider(len(faces[root])))
        for shape in shapes:
            # The mapping generator includes both cyclic directions, so the
            # reversed cycle traversal would enumerate the same vertex maps.
            for cycle in face_cycles[root][:1]:
                for mapping in _atom_to_vertex_maps(cycle, shape):
                    _charge_embedding_work(work)
                    coords = {atom: shape.vertices[vertex] for atom, vertex in mapping.items()}
                    placed = {root: (shape, 0.0, 0.0)}
                    _place_tree(mol, faces, face_cycles, face_adjacency, face_edges,
                                root, placed, coords, raw, work, shape_provider,
                                allow_shared_carbon_junctions or allow_peri_fused, face_shapes)
    distorted = False
    try:
        if (polynaphthylene_core and _traversal_root is None
                and not allow_shared_carbon_junctions and not allow_peri_fused):
            raw.extend(_regular_arm_embeddings(faces, face_cycles, face_adjacency, face_edges))
        else:
            search()
        if not raw and allow_larger_system:
            # P-25.3.2.3.2 calls for a smallest ring to be distorted. Search
            # every face of that size and retain ties; atom and face order
            # never decide which candidate drawing is selected.
            smallest = min(map(len, faces))
            for index, face in enumerate(faces):
                if len(face) != smallest:
                    continue
                neighbors = tuple(len(faces[j]) for j in face_adjacency[index])
                shapes = _edge_matched_distortions(smallest, neighbors, shape_provider)
                if shapes:
                    search({index: shapes})
            distorted = bool(raw)
    except _EmbeddingSearchLimit:
        # Never rank a truncated set: a later embedding could win a source
        # criterion or introduce a tie.
        return FusedOrientationFrontier(False, "bounded embedding search exceeded", graph)
    if not raw:
        return FusedOrientationFrontier(False, "no overlap-free embedding from the illustrated shapes", graph)

    unique = {}
    for item in raw:
        # Atom IDs identify mapped graph vertices here; they do not rank or
        # select drawings. Deduplication is by exact graph-to-coordinate map.
        key = tuple(sorted((atom, round(x, 7), round(y, 7))
                           for atom, (x, y) in item["coords"].items()))
        unique.setdefault(key, item)

    # FR-5.1.6: fusion at elongated bonds is used only when ordinary
    # source-shape bonds cannot construct the system, and then as few such
    # bonds as possible. Apply that drawing criterion before orientation.
    def elongated_fusions(item):
        coords = item['coords']
        return sum(math.dist(coords[a], coords[b]) > 1.0 + _EPS
                   for edge in graph.fusion_edges for a, b in [tuple(edge)])
    fewest_elongated = min(map(elongated_fusions, unique.values()))
    unique = {key: item for key, item in unique.items()
              if elongated_fusions(item) == fewest_elongated}

    branching = any(len(neighbors) > 2 for neighbors in face_adjacency.values())
    evaluated = []
    longest_row = 0
    for item in unique.values():
        face_centers = {i: _placement_row_axis_center(placement)
                        for i, placement in item["placed"].items()}
        row_options = _horizontal_rows(item["coords"], face_centers, faces,
                                       face_adjacency, face_edges,
                                       row_eligible=frozenset(i for i, placement in item['placed'].items()
                                           if not placement[0].variant.startswith('pentagon_side_')))
        if not row_options or len(row_options[0][0]) < longest_row:
            continue
        if len(row_options[0][0]) > longest_row:
            longest_row = len(row_options[0][0])
            evaluated.clear()  # Row length precedes every quadrant criterion.
        quadrants = {}
        for row, extent, center in row_options:
            if center not in quadrants:
                quadrants[center] = _quadrant_counts(item["coords"], face_centers, faces, center)
            ur, ll, above = quadrants[center]
            # P-25.3.2.3.3(a-d) ranks row count, upper-right, lower-left,
            # then above-row counts. The rule's furthest-apart vertical-bond
            # instruction identifies the row's bonds; it is metadata, not an
            # additional width criterion between candidate orientations.
            score = _orientation_criterion_key(len(row), ur, ll, above)
            # Do not merge disconnected collinear runs into one row without a
            # source rule for that arrangement. Keep the issue attached to
            # this drawing until all P-25 orientation criteria are applied.
            disconnected_collinear = branching and any(
                i not in row and math.isclose(face_centers[i][1], center[1], abs_tol=_EPS)
                for i in range(len(faces)))
            evaluated.append((score, row, extent, ur, ll, above,
                              disconnected_collinear, item, center))
    if not evaluated:
        return FusedOrientationFrontier(False, "no eligible connected horizontal row", graph)

    winners = _source_orientation_winners(evaluated)
    if proved_five_seven_six and any(len(row[1]) != 3 for row in winners):
        return FusedOrientationFrontier(False, "5/7/6 full-row orientation unproved", graph)
    if any(row[6] for row in winners):
        return FusedOrientationFrontier(
            False, "winning drawing has disconnected collinear ring runs outside scope", graph)
    winning_orders = _complete_winner_numberings(
        winners,
        lambda record: _clockwise_perimeter_orders(
            mol, graph, faces,
            {i: _placement_row_axis_center(placement)
             for i, placement in record[7]["placed"].items()},
            record[7]["coords"], record[1], record[8],
            signed_grid_steps=polynaphthylene_core,
            ideal_hex_grid=all(placement[0].variant.startswith(('pentagon_', 'hexagon_'))
                               for placement in record[7]['placed'].values())))
    if winning_orders is None:
        return FusedOrientationFrontier(
            False, "winning drawing lacks a proved uppermost-ring perimeter start", graph)
    candidates = []
    for (score, row, extent, ur, ll, above, _, item, center), orders in zip(winners, winning_orders):
        candidate_face_centers = {i: _placement_row_axis_center(placement)
                                  for i, placement in item["placed"].items()}
        face_shape_facts = tuple(sorted(
            (tuple(sorted(faces[i])), item["placed"][i][0].variant)
            for i in range(len(faces))))
        for seed in orders:
            candidates.append(GeometricEmbedding(
                tuple(sorted((atom, float(x), float(y)) for atom, (x, y) in item["coords"].items())),
                face_shape_facts,
                tuple(sorted((tuple(sorted(faces[i])), float(candidate_face_centers[i][0]),
                              float(candidate_face_centers[i][1])) for i in range(len(faces)))),
                tuple(frozenset(faces[i]) for i in row),
                len(row), float(extent), float(ur), float(ll), float(above), score,
                seed.face_path, seed.transition_fusion_edges,
                seed.successor_perimeter_edges, seed.clockwise_order))
    # Equivalent graph-to-locant seeds are reported once; unresolved geometric
    # ties remain visible as multiple seeds.
    drawing_source = ("; edge-matched smallest-ring distortion" if distorted
                      else "; finite illustrated-shape embedding")
    seeds_by_order = {item.clockwise_order: PeripheralOrientationSeed(
        item.clockwise_order, _ROOT_SOURCE + drawing_source)
        for item in candidates}
    return FusedOrientationFrontier(True, None, graph, tuple(candidates),
                                    tuple(seeds_by_order.values()))


def _bounded_five_seven_six_oxygen_tree(mol, faces):
    """Admit the tested C/O tree where O bridges the two shared edges."""
    five, six, seven = (next(face for face in faces if len(face) == size) for size in (5, 6, 7))
    if five & six or len(five & seven) != 2 or len(six & seven) != 2:
        return False
    core = five | six | seven
    if (sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in core) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in core)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in five | six)):
        return False
    oxygen = next(i for i in seven if mol.GetAtomWithIdx(i).GetAtomicNum() == 8)
    neighbors = {a.GetIdx() for a in mol.GetAtomWithIdx(oxygen).GetNeighbors()}
    return len(neighbors & (five & seven)) == len(neighbors & (six & seven)) == 1


def _orientation_criterion_key(row_count, upper_right, lower_left, above_row):
    """P-25.3.2.3.3(a-d) order; row width is descriptive metadata only."""
    return (float(row_count), round(float(upper_right), 7),
            -round(float(lower_left), 7), round(float(above_row), 7))


def _source_orientation_winners(records):
    """Retain every source-tied record after applying only P-25.3.2.3.3(a-d)."""
    best = max(record[0] for record in records)
    return [record for record in records if record[0] == best]


def _complete_winner_numberings(winners, numbering_for):
    """Number only after orientation is fixed; fail if any winner is unresolved."""
    orders = [tuple(numbering_for(record)) for record in winners]
    if not orders or any(not order_set for order_set in orders):
        return None
    return orders


def _shapes(size):
    """Coordinates are finite schematic templates from P-25.3.2.3.1 Figs 583–589."""
    if size == 3:
        return (_Shape(3, "triangle_point_right",
                       ((-0.5, -0.5), (math.sqrt(3) / 2 - 0.5, 0.0), (-0.5, 0.5))),
                _Shape(3, "triangle_point_left",
                       ((0.5, -0.5), (0.5 - math.sqrt(3) / 2, 0.0), (0.5, 0.5))))
    if size == 4:
        return (_Shape(4, "square", ((-0.5, -0.5), (0.5, -0.5),
                                      (0.5, 0.5), (-0.5, 0.5))),)
    if size == 5:
        # P-25.3.2.3.1/FR-5.1.1: a five-membered row ring is a
        # six-membered outline with one vertex removed. Its ordinary sides
        # therefore coincide with the hexagon's vertical and diagonal sides;
        # only the edge replacing the missing vertex is elongated.
        half_width = math.sqrt(3) / 2
        up = ((-half_width, -0.5), (half_width, -0.5), (half_width, 0.5),
              (0.0, 1.0), (-half_width, 0.5))
        down = tuple((x, -y) for x, y in up)
        return (_Shape(5, "pentagon_point_up", up),
                _Shape(5, "pentagon_point_down", down))
    if size == 6:
        h = math.sqrt(3) / 2
        return (_Shape(6, "hexagon_point_up",
                       ((0.0, -1.0), (h, -0.5), (h, 0.5),
                        (0.0, 1.0), (-h, 0.5), (-h, -0.5))),)
    if size == 7:
        # Preserve the established finite-template path for scopes without
        # the separately proved five/seven/six oxygen-junction construction.
        radius = 1 / (2 * math.sin(math.pi / 7))
        flat_top = tuple((radius * math.cos(math.radians(90 - 180 / 7 + 360 * i / 7)),
                          radius * math.sin(math.radians(90 - 180 / 7 + 360 * i / 7)))
                         for i in range(7))
        point_top = tuple((radius * math.cos(math.radians(90 + 360 * i / 7)),
                           radius * math.sin(math.radians(90 + 360 * i / 7)))
                          for i in range(7))
        return (_Shape(7, "heptagon_flat_top", flat_top),
                _Shape(7, "heptagon_point_top", point_top))
    if size == 8:
        # Figures 588–589 show octagon outlines with edge- and vertex-centered
        # horizontal-axis phases. The marker details in Figure 589 are not used
        # as atom or fusion-edge labels by this geometry-only model.
        radius = 1 / (2 * math.sin(math.pi / 8))
        edge_top = tuple((radius * math.cos(math.radians(67.5 + 45 * i)),
                          radius * math.sin(math.radians(67.5 + 45 * i)))
                         for i in range(8))
        point_top = tuple((radius * math.cos(math.radians(90 + 45 * i)),
                           radius * math.sin(math.radians(90 + 45 * i)))
                          for i in range(8))
        return (_Shape(8, "octagon_edge_top", edge_top),
                _Shape(8, "octagon_point_top", point_top))
    return ()


@lru_cache(maxsize=3)
def _polynaphthylene_shapes(size):
    # FR-5.1.1 inserts pairs of horizontal bonds into a hexagonal outline.
    # Off-row hexagons can also fuse to a horizontal edge (FR-5.1.2).
    if size == 6:
        upright, = _shapes(6)
        return upright, _Shape(6, 'hexagon_point_right', tuple((-y, x) for x, y in upright.vertices))
    count = (size - 6) // 2
    width = math.sqrt(3) / 2 + count / 2
    horizontal = tuple(i - count / 2 for i in range(count + 1))
    points = ((-width, -0.5), *((x, -1.) for x in horizontal), (width, -0.5),
              (width, 0.5), *((x, 1.) for x in reversed(horizontal)), (-width, 0.5))
    standard = _Shape(size, 'horizontal_insert', points)
    if size == 10:
        # FR-5.1.2 also allows the transversely extended decagon when
        # additional rings fuse above/below. Only its central vertical
        # bonds lie on the bisector shared by the horizontal-row centres.
        return standard, _Shape(size, 'vertical_insert', tuple((-y, x) for x, y in points))
    return (standard,)


@lru_cache(maxsize=1)
def _hexagons_by_directed_edge():
    """Index both regular hexagon templates by fusion vector and side of the edge."""
    result = {}
    for shape in _polynaphthylene_shapes(6):
        for start, (ax, ay) in enumerate(shape.vertices):
            for step in (1, -1):
                offsets = tuple((shape.vertices[(start + k * step) % 6][0] - ax,
                                 shape.vertices[(start + k * step) % 6][1] - ay) for k in range(6))
                dx, dy = offsets[1]
                side = dx * -ay - dy * -ax > 0
                result[round(dx, 7), round(dy, 7), side] = (shape, -ax, -ay, offsets[2:])
    return result


def _regular_arm_embeddings(faces, cycles, adjacency, fusion_edges):
    """A central polygon's orientation fixes every exterior regular hexagon in its face tree."""
    center = next(i for i, face in enumerate(faces) if len(face) != 6)
    order = [center]
    joins = []
    for parent in order:
        for child in sorted(adjacency[parent].difference(order)):
            edge = _face_adjacent(parent, child, fusion_edges)
            cycle = cycles[child][0]
            start = next(i for i, a in enumerate(cycle) if a in edge and cycle[(i + 1) % 6] in edge)
            a, b, *others = cycle[start:] + cycle[:start]
            joins.append((parent, child, a, b, others))
            order.append(child)
    hexagons = _hexagons_by_directed_edge()
    disjoint_pairs = [(i, j) for i in range(len(faces)) for j in range(i) if j not in adjacency[i]]
    for shape in _polynaphthylene_shapes(len(faces[center])):
        for mapping in _atom_to_vertex_maps(cycles[center][0], shape):
            coords = {i: shape.vertices[v] for i, v in mapping.items()}
            placed = {center: (shape, 0., 0.)}
            for parent, child, a, b, others in joins:
                ax, ay = coords[a]
                bx, by = coords[b]
                dx, dy = bx - ax, by - ay
                # All templates here are centrally symmetric about their origin.
                _, px, py = placed[parent]
                outside = dx * (py - ay) - dy * (px - ax) < 0
                hexagon, tx, ty, offsets = hexagons[round(dx, 7), round(dy, 7), outside]
                placed[child] = (hexagon, ax + tx, ay + ty)
                coords.update((i, (ax + x, ay + y)) for i, (x, y) in zip(others, offsets))
            # The two convex polygons at each fusion were placed on opposite
            # sides of their shared bond. Other face pairs must be disjoint.
            if all(_separate_arm_faces(placed[i], placed[j]) for i, j in disjoint_pairs):
                yield {'coords': coords, 'placed': placed}


def _separate_arm_faces(left, right):
    shape, x, y = left
    other, u, v = right
    # The largest template radius is less than two bond lengths.
    if abs(u - x) > 4 or abs(v - y) > 4:
        return True
    return _separate_translated_polygons(shape.vertices, other.vertices, round(u - x, 10), round(v - y, 10))


@lru_cache(maxsize=2048)
def _separate_translated_polygons(left, right, dx, dy):
    return _polygons_nonoverlapping(left, tuple((x + dx, y + dy) for x, y in right), None)


def _seven_fused_shapes(size):
    if size != 7:
        return _shapes(size)
    # Figure 587 shows vertical left and right edges, unlike the existing
    # regular-heptagon templates. Normalize all edges to unit length while
    # retaining the source's top-edge and top-vertex phases.
    half_width = math.sqrt(3) / 2
    cap_y = 0.5 + math.sqrt(1 - (half_width - 0.5) ** 2)
    flat_top = ((-0.5, cap_y), (0.5, cap_y), (half_width, 0.5),
                (half_width, -0.5), (0.0, -1.0),
                (-half_width, -0.5), (-half_width, 0.5))
    point_top = tuple((x, -y) for x, y in flat_top)
    return (_Shape(7, "heptagon_flat_top", flat_top),
            _Shape(7, "heptagon_point_top", point_top))


def _nonrow_pentagon_shapes():
    """FR-5.1.1/.2: two row forms and four additional nonrow forms.

    Each is a hexagonal grid cell with one vertex deleted. The four sloping
    elongated edges are allowed outside the main row, not as row drawings.
    """
    ordinary = _shapes(5)
    result = list(ordinary)
    for phase in (1, 2, 4, 5):
        angle = math.radians(phase * 60)
        cosine, sine = math.cos(angle), math.sin(angle)
        vertices = tuple((x*cosine-y*sine, x*sine+y*cosine)
                         for x, y in ordinary[0].vertices)
        result.append(_Shape(5, f'pentagon_side_{phase}', vertices))
    return tuple(result)


def _edge_matched_distortions(size, neighbor_sizes, shape_provider):
    """Derive axis-preserving distortions from possible shared-edge vectors.

    A nonzero x or y component fixes the corresponding scale from the two
    templates. If both components are zero, that axis is left unchanged.
    This finite construction is not a claim to cover arbitrary distortions.
    """
    def vectors(shape):
        vertices = shape.vertices
        return tuple((abs(b[0] - a[0]), abs(b[1] - a[1]))
                     for a, b in zip(vertices, vertices[1:] + vertices[:1]))

    neighboring = tuple(vector for neighbor_size in set(neighbor_sizes)
                        for shape in shape_provider(neighbor_size)
                        for vector in vectors(shape))
    result = {}
    for shape in shape_provider(size):
        for dx, dy in vectors(shape):
            for tx, ty in neighboring:
                if ((dx < _EPS) != (tx < _EPS) or
                        (dy < _EPS) != (ty < _EPS)):
                    continue
                sx = tx / dx if dx >= _EPS else 1.0
                sy = ty / dy if dy >= _EPS else 1.0
                if math.isclose(sx, 1.0, abs_tol=_EPS) and math.isclose(sy, 1.0, abs_tol=_EPS):
                    continue
                key = (shape.variant, round(sx, 7), round(sy, 7))
                result.setdefault(key, _Shape(
                    size, "distorted_" + shape.variant,
                    tuple((x * sx, y * sy) for x, y in shape.vertices)))
    if size == 4 and len(neighbor_sizes) >= 2:
        # A four-membered peri-fused ring can need a skew rather than an
        # axis scaling. Its two edge directions can be fixed by compatible
        # edges of the adjacent undistorted faces. The square is centered at
        # the origin, so these linear transforms preserve its row center.
        from itertools import product

        directions = {}
        for neighbor_size in set(neighbor_sizes):
            for shape in shape_provider(neighbor_size):
                for a, b in zip(shape.vertices, shape.vertices[1:] + shape.vertices[:1]):
                    vector = (b[0]-a[0], b[1]-a[1])
                    directions[(round(vector[0], 7), round(vector[1], 7))] = vector
        for horizontal, vertical in product(directions.values(), repeat=2):
            if _cross(horizontal, vertical) <= _EPS:
                continue
            vertices = tuple((horizontal[0]*x + vertical[0]*y,
                              horizontal[1]*x + vertical[1]*y)
                             for x, y in _shapes(4)[0].vertices)
            # Vertex rotations/reversals are covered by the atom mappings.
            key = ('affine-square', tuple(sorted((round(x, 7), round(y, 7))
                                                  for x, y in vertices)))
            result.setdefault(key, _Shape(4, 'distorted_affine_square', vertices))
    return tuple(result.values())


def _face_cycle_orders(mol, face):
    adjacency = {i: {a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in face}
                 for i in face}
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return ()
    start = min(face)
    first = min(adjacency[start])
    cycle = [start, first]
    while len(cycle) < len(face):
        options = adjacency[cycle[-1]] - {cycle[-2]}
        if len(options) != 1:
            return ()
        cycle.append(next(iter(options)))
    if start not in adjacency[cycle[-1]]:
        return ()
    return (tuple(cycle), tuple((cycle[0],) + tuple(reversed(cycle[1:]))))


def _atom_to_vertex_maps(cycle, shape):
    n = len(cycle)
    result = []
    for offset in range(n):
        for direction in (1, -1):
            result.append({cycle[k]: (offset + direction * k) % n for k in range(n)})
    return result


def _face_adjacent(face_i, face_j, face_edges):
    key = (min(face_i, face_j), max(face_i, face_j))
    return face_edges.get(key)


def _place_tree(mol, faces, face_cycles, face_adjacency, face_edges,
                root, placed, coords, raw, work, shape_provider,
                allow_shared_carbon_junctions=False, face_shapes=None):
    if len(placed) == len(faces):
        if not _all_faces_nonoverlapping(faces, coords, face_edges, allow_shared_carbon_junctions):
            return False
        if len(raw) >= _MAX_EMBEDDINGS:
            raise _EmbeddingSearchLimit
        raw.append({"coords": dict(coords), "placed": dict(placed)})
        return True
    parent = next((i for i in placed if any(j not in placed for j in face_adjacency[i])), None)
    if parent is None:
        return False
    # Every completed embedding includes every not-yet-placed neighbor.
    # Choosing a single next neighbor only fixes the traversal order: all of
    # that child's shapes and vertex mappings are still explored below.
    # Trying the other neighbor orders repeats the same complete drawings,
    # factorially so for branched and peri-fused face graphs.
    for child in sorted(face_adjacency[parent] - set(placed))[:1]:
        edge = _face_adjacent(parent, child, face_edges)
        if edge is None:
            continue
        shared = tuple(edge)
        shapes = (face_shapes[child] if face_shapes and child in face_shapes
                  else shape_provider(len(faces[child])))
        for shape in shapes:
            # `_atom_to_vertex_maps` already covers both directions; taking
            # the second cycle representation would duplicate this frontier.
            for cycle in face_cycles[child][:1]:
                for mapping in _atom_to_vertex_maps(cycle, shape):
                    _charge_embedding_work(work)
                    # The vertex map already fixes both ends of the shared
                    # edge. Swapping them constructs the same translation;
                    # directions of the child cycle are already enumerated
                    # by _atom_to_vertex_maps.
                    for a, b in (shared,):
                        if a not in coords or b not in coords:
                            continue
                        va, vb = mapping[a], mapping[b]
                        pa, pb = shape.vertices[va], shape.vertices[vb]
                        # Catalog axes and phase stay fixed. The only free
                        # placement parameter is this translation; introducing
                        # a child rotation would invalidate leaf-root
                        # completeness.
                        tx, ty = coords[a][0] - pa[0], coords[a][1] - pa[1]
                        predicted_b = (pb[0] + tx, pb[1] + ty)
                        if not _close_point(predicted_b, coords[b]):
                            continue
                        trial = dict(coords)
                        for atom, vertex in mapping.items():
                            point = (shape.vertices[vertex][0] + tx,
                                     shape.vertices[vertex][1] + ty)
                            if atom in trial and not _close_point(trial[atom], point):
                                break
                            if _vertex_collision(trial, atom, point):
                                break
                            trial[atom] = point
                        else:
                            if not _opposite_sides(faces[parent], faces[child], edge, coords, trial):
                                continue
                            next_placed = dict(placed)
                            next_placed[child] = (shape, tx, ty)
                            # Earlier face pairs were checked when the later
                            # member was placed. Only pairs involving this
                            # new face can introduce an overlap now.
                            if any(not _all_faces_nonoverlapping(
                                    (faces[i], faces[child]), trial, face_edges,
                                    allow_shared_carbon_junctions) for i in placed):
                                continue
                            _place_tree(mol, faces, face_cycles, face_adjacency, face_edges,
                                        root, next_placed, trial, raw, work, shape_provider,
                                        allow_shared_carbon_junctions, face_shapes)
    return bool(raw)


def _all_faces_nonoverlapping(faces, coords, face_edges, allow_shared_carbon_junctions=False):
    if any(atom not in coords for face in faces for atom in face):
        return False
    polygons = [None] * len(faces)
    for i, face in enumerate(faces):
        for j in range(i + 1, len(faces)):
            shared = face & faces[j]
            if len(shared) == 2:
                edge = frozenset(shared)
                # Sharing two atoms is a permitted fusion only when those
                # atoms form the recorded common bond, and the face interiors
                # lie on opposite sides of it.
                if edge not in face_edges.values():
                    return False
                if not _opposite_sides(face, faces[j], edge, coords, coords):
                    return False
                continue
            if shared and not (allow_shared_carbon_junctions and len(shared) == 1):
                return False
            # Shared-edge pairs are settled above. Order each polygon only
            # when a pair needs the point and segment tests.
            if polygons[i] is None:
                polygons[i] = _order_face_polygon(tuple(coords[atom] for atom in face))
            if polygons[j] is None:
                polygons[j] = _order_face_polygon(tuple(coords[atom] for atom in faces[j]))
            touch = coords[next(iter(shared))] if shared else None
            if not _polygons_nonoverlapping(polygons[i], polygons[j], touch):
                return False
    return True


@lru_cache(maxsize=32768)
def _polygons_nonoverlapping(polygon_i, polygon_j, touch):
    """Reuse a geometric verdict for these coordinates and permitted contact."""
    for geometry, other in ((polygon_i, polygon_j), (polygon_j, polygon_i)):
        for point in geometry:
            if (touch is None or not _close_point(point, touch)) and _point_inside(point, other):
                return False
    # Bounding boxes can exclude segment intersections using the existing
    # segment tolerance. Point-in-polygon tests retain their own tolerance.
    min_x_i, max_x_i, min_y_i, max_y_i = _polygon_bounds(polygon_i)
    min_x_j, max_x_j, min_y_j, max_y_j = _polygon_bounds(polygon_j)
    if (max_x_i + _EPS < min_x_j or max_x_j + _EPS < min_x_i
            or max_y_i + _EPS < min_y_j or max_y_j + _EPS < min_y_i):
        return True
    for a, b in _polygon_segments(polygon_i):
        for c, d in _polygon_segments(polygon_j):
            if not _segments_intersect_or_touch(a, b, c, d):
                continue
            if (touch is None or not any(_close_point(p, touch) for p in (a, b))
                    or not any(_close_point(p, touch) for p in (c, d))):
                return False
            first = b if _close_point(a, touch) else a
            second = d if _close_point(c, touch) else c
            u = (first[0]-touch[0], first[1]-touch[1])
            v = (second[0]-touch[0], second[1]-touch[1])
            if abs(_cross(u, v)) <= _EPS and u[0]*v[0]+u[1]*v[1] > _EPS:
                return False  # The two edges overlap along the same ray.
    return True


@lru_cache(maxsize=8192)
def _order_face_polygon(points):
    # Convex face coordinates are recovered by angle about their centroid;
    # this ordering is geometric and has no atom-ID ranking role.
    center = (sum(p[0] for p in points) / len(points),
              sum(p[1] for p in points) / len(points))
    return tuple(sorted(points,
                        key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0])))


@lru_cache(maxsize=8192)
def _polygon_bounds(poly):
    return (min(p[0] for p in poly), max(p[0] for p in poly),
            min(p[1] for p in poly), max(p[1] for p in poly))


@lru_cache(maxsize=4096)
def _polygon_segments(poly):
    return tuple(zip(poly, poly[1:] + poly[:1]))


@lru_cache(maxsize=4096)
def _polygon_edge_vectors(poly):
    return tuple(
        (a[0], a[1], b[0] - a[0], b[1] - a[1])
        for a, b in _polygon_segments(poly)
    )


def _point_inside(point, poly):
    x, y = point
    side = None
    for start_x, start_y, edge_x, edge_y in _polygon_edge_vectors(poly):
        cross = edge_x * (y - start_y) - edge_y * (x - start_x)
        if abs(cross) > _EPS:
            sign = cross > 0
            if side is not None and sign != side:
                return False
            side = sign
    return side is not None


def _segments_intersect_or_touch(a, b, c, d):
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    ab_x, ab_y = bx - ax, by - ay
    cd_x, cd_y = dx - cx, dy - cy
    ab_c = ab_x * (cy - ay) - ab_y * (cx - ax)
    ab_d = ab_x * (dy - ay) - ab_y * (dx - ax)
    cd_a = cd_x * (ay - cy) - cd_y * (ax - cx)
    cd_b = cd_x * (by - cy) - cd_y * (bx - cx)
    if ab_c * ab_d < -_EPS and cd_a * cd_b < -_EPS:
        return True
    return (_point_on_segment(c, a, b, ab_c) or _point_on_segment(d, a, b, ab_d)
            or _point_on_segment(a, c, d, cd_a) or _point_on_segment(b, c, d, cd_b))


def _point_on_segment(point, start, end, cross):
    if abs(cross) > _EPS:
        return False
    return (min(start[0], end[0]) - _EPS <= point[0] <= max(start[0], end[0]) + _EPS
            and min(start[1], end[1]) - _EPS <= point[1] <= max(start[1], end[1]) + _EPS)


def _opposite_sides(face_a, face_b, edge, coords_a, coords_b):
    a, b = tuple(edge)
    if a not in coords_a or b not in coords_a or a not in coords_b or b not in coords_b:
        return False
    pa, pb = coords_a[a], coords_a[b]
    ca = _centroid(tuple(coords_a[i] for i in face_a))
    cb = _centroid(tuple(coords_b[i] for i in face_b))
    side_a = _cross((pb[0] - pa[0], pb[1] - pa[1]), (ca[0] - pa[0], ca[1] - pa[1]))
    side_b = _cross((pb[0] - pa[0], pb[1] - pa[1]), (cb[0] - pa[0], cb[1] - pa[1]))
    return side_a * side_b < -_EPS


def _centroid(points):
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def _placement_row_axis_center(placement):
    shape, tx, ty = placement
    dx, dy = shape.row_axis_center
    return (tx + dx, ty + dy)


def _charge_embedding_work(work):
    work[0] += 1
    if work[0] > _MAX_EMBEDDING_WORK:
        raise _EmbeddingSearchLimit


def _clockwise_nonfusion_block_start(eligible, clockwise_ring_cycle):
    """Return the first atom only when there is exactly one nonfusion block.

    P-25.3.3.1.1's most-counterclockwise nonfusion atom is the atom at the
    counterclockwise end of the peripheral segment. In the prescribed drawing,
    when the ring cycle is traversed clockwise, this is the first nonfusion
    atom after a fusion atom. This convenience helper deliberately reports
    only a unique block; callers handling multiple blocks use the block list
    and the source-supported uppermost-start tie-break below.
    """
    eligible = frozenset(eligible)
    cycle = tuple(clockwise_ring_cycle)
    if not eligible or len(cycle) < 3 or len(set(cycle)) != len(cycle):
        return None
    if not eligible.issubset(cycle):
        return None
    starts = [atom for previous, atom in zip(cycle[-1:] + cycle[:-1], cycle)
              if atom in eligible and previous not in eligible]
    return starts[0] if len(starts) == 1 else None


def _clockwise_nonfusion_block_starts(eligible, clockwise_ring_cycle):
    """Return each maximal nonfusion run's clockwise boundary atom.

    The 1998 IUPAC FR-5.3 text clarifies that a choice is made within a
    continuous nonfusion portion and that any remaining choice goes to the
    uppermost start. For a clockwise ring cycle, each portion's counterclockwise
    boundary is its first eligible atom after a fusion atom.
    """
    eligible = frozenset(eligible)
    cycle = tuple(clockwise_ring_cycle)
    if not eligible or len(cycle) < 3 or len(set(cycle)) != len(cycle):
        return ()
    if not eligible.issubset(cycle):
        return ()
    return tuple(atom for previous, atom in zip(cycle[-1:] + cycle[:-1], cycle)
                 if atom in eligible and previous not in eligible)


def _uppermost_nonfusion_block_starts(eligible, clockwise_ring_cycle, coords):
    """Choose each block's clockwise boundary, then apply uppermost tie-break."""
    starts = _clockwise_nonfusion_block_starts(eligible, clockwise_ring_cycle)
    if not starts:
        return ()
    try:
        highest_y = max(coords[atom][1] for atom in starts)
    except (KeyError, TypeError, IndexError):
        return ()
    return tuple(atom for atom in starts
                 if math.isclose(coords[atom][1], highest_y, abs_tol=_EPS))


def _horizontal_rows(coords, face_centers, faces, adjacency, face_edges, *, row_eligible=None):
    rows = []
    for start in range(len(faces)):
        if row_eligible is not None and start not in row_eligible:
            continue
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            edge_centers = []
            valid = True
            for a, b in zip(path, path[1:]):
                edge = _face_adjacent(a, b, face_edges)
                u, v = tuple(edge)
                if not math.isclose(coords[u][0], coords[v][0], abs_tol=_EPS):
                    valid = False
                    break
                edge_centers.append((coords[u][0], (coords[u][1] + coords[v][1]) / 2))
            if not valid:
                continue
            ys = [face_centers[i][1] for i in path]
            if any(not math.isclose(y, ys[0], abs_tol=_EPS) for y in ys[1:]):
                continue
            if len(path) == 1 or len(set(path)) == len(path):
                center = _row_center(path, faces, face_centers, coords, face_edges)
                if center is not None:
                    # The source says the relevant vertical common bonds are
                    # those furthest apart; row extent is their x-span.
                    xs = [x for x, _ in edge_centers]
                    extent = max(xs) - min(xs) if len(xs) > 1 else 0.0
                    rows.append((path, extent, center))
            for nxt in adjacency[node] - set(path):
                if row_eligible is not None and nxt not in row_eligible:
                    continue
                if len(path) < len(faces):
                    stack.append((nxt, path + (nxt,)))
    max_count = max((len(row[0]) for row in rows), default=1)
    rows = [row for row in rows if len(row[0]) == max_count]
    if row_eligible is not None:
        return rows  # No row may be invented from nonrow-only shapes.
    return rows or [((0,), 0.0, _centroid(tuple(coords[i] for i in faces[0])))]


def _row_center(path, faces, face_centers, coords, face_edges):
    if len(path) % 2:
        middle = path[len(path) // 2]
        return face_centers[middle]
    left, right = path[len(path) // 2 - 1:len(path) // 2 + 1]
    edge = _face_adjacent(left, right, face_edges)
    if edge is None:
        return None
    return _centroid(tuple(coords[i] for i in edge))


def _quadrant_counts(coords, face_centers, faces, center):
    cx, cy = center
    upper_right = lower_left = above = 0.0
    for i, face in enumerate(faces):
        points = tuple(coords[i] for i in face)
        min_x, max_x = min(p[0] for p in points), max(p[0] for p in points)
        min_y, max_y = min(p[1] for p in points), max(p[1] for p in points)
        cross_x = min_x < cx - _EPS and max_x > cx + _EPS
        cross_y = min_y < cy - _EPS and max_y > cy + _EPS
        center_x, center_y = face_centers[i]
        if cross_x and cross_y:
            upper_right += 0.25
            lower_left += 0.25
            above += 0.5
        elif cross_x:
            if center_y > cy:
                upper_right += 0.5
                # The ring is wholly above the horizontal row. Its split by
                # the vertical axis affects quadrant counts, not criterion d.
                above += 1.0
            elif center_y < cy:
                lower_left += 0.5
        elif cross_y:
            if center_x > cx:
                upper_right += 0.5
            elif center_x < cx:
                lower_left += 0.5
            above += 0.5
        else:
            if center_x > cx and center_y > cy:
                upper_right += 1.0
                above += 1.0
            elif center_x < cx and center_y < cy:
                lower_left += 1.0
            elif center_y > cy:
                above += 1.0
    return upper_right, lower_left, above


def _clockwise_perimeter_orders(mol, graph, faces, face_centers, coords,
                                horizontal_row, row_center, *, ideal_hex_grid=False, signed_grid_steps=False):
    """Choose uppermost/rightmost ring by source row levels, then start.

    FR-5.3 says that differently sized rings are ranked by their level in the
    idealized grid above the main horizontal row, rather than raw geometric
    height. This mirrors that statement with face-tree shells. Above-side
    faces that cannot be connected to the selected row through above-side
    faces are left unresolved.
    """
    shared_atoms = set().union(*(edge for edge in graph.fusion_edges))
    row = frozenset(horizontal_row)
    if not row or not row.issubset(range(len(faces))):
        return ()
    row_y = row_center[1]
    above = set()
    for index in range(len(faces)):
        y = face_centers[index][1]
        if index in row:
            if not math.isclose(y, row_y, abs_tol=_EPS):
                return ()
        elif math.isclose(y, row_y, abs_tol=_EPS):
            # An unclassified face on the row axis is an overlapping/disconnected
            # row ambiguity, not evidence for either upper or lower level.
            return ()
        elif y > row_y:
            above.add(index)

    adjacency = {i: set() for i in range(len(faces))}
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if len(faces[i] & faces[j]) == 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    levels = {index: 0 for index in row}
    queue = list(row)
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency[current] & above:
            if neighbor not in levels:
                # FR-5.3 compares levels in an idealised grid. A sideways
                # fusion on an off-row arm stays at its current level. An
                # inward bend can equally step back down toward the row.
                step = 1
                if signed_grid_steps:
                    dy = face_centers[neighbor][1] - face_centers[current][1]
                    step = int(dy > _EPS) - int(dy < -_EPS)
                levels[neighbor] = levels[current] + step
                queue.append(neighbor)
    if above - set(levels):
        return ()
    if ideal_hex_grid or len(set(map(len, faces))) == 1:
        # FR-5.3's idealized-grid qualification concerns differently sized
        # rings. Equal-sized rings use the actual uppermost row. Graph
        # distance from the main row can tie rings at different heights.
        # The six deleted-vertex pentagon forms supply their original
        # hexagonal cell centres, so their grid heights are proved too.
        top = max(face_centers[i][1] for i in range(len(faces)))
        highest = [i for i in range(len(faces))
                   if math.isclose(face_centers[i][1], top, abs_tol=_EPS)]
    else:
        max_level = max(levels.values(), default=0)
        highest = [i for i, level in levels.items() if level == max_level]
    rightmost_x = max(face_centers[i][0] for i in highest)
    selected_faces = [i for i in highest
                      if math.isclose(face_centers[i][0], rightmost_x, abs_tol=_EPS)]
    peripheral_atoms = set().union(*graph.perimeter_edges)
    perimeter = {i: set() for i in peripheral_atoms}
    for edge in graph.perimeter_edges:
        a, b = tuple(edge)
        perimeter[a].add(b)
        perimeter[b].add(a)
    orders = []
    for selected_index in selected_faces:
        selected = faces[selected_index]
        eligible = [atom for atom in selected if atom not in shared_atoms]
        face_path = [selected_index]
        transition_edges = []
        successor_edges = []
        continuation_position = None
        visited = {selected_index}
        while not eligible:
            successor = _clockwise_successor_face(
                graph, faces, mol, coords, face_path[-1],
                from_position=continuation_position)
            if successor is None or successor[0] in visited:
                return ()
            next_face, perimeter_edge, continuation_position = successor
            shared = faces[face_path[-1]] & faces[next_face]
            if len(shared) != 2:
                return ()
            face_path.append(next_face)
            transition_edges.append(tuple(sorted(shared)))
            successor_edges.append(tuple(sorted(perimeter_edge)))
            visited.add(next_face)
            eligible = [atom for atom in faces[next_face] if atom not in shared_atoms]
        selected_for_start = faces[face_path[-1]]
        cycle = next((order for order in _face_cycle_orders(mol, selected_for_start)
                      if _signed_area(tuple(coords[i] for i in order)) < -_EPS), None)
        if cycle is None:
            continue
        # Each nonfusion block contributes its counterclockwise boundary.
        # FR-5.3's corroborating tie-break selects the uppermost such atom;
        # exact height ties remain visible as alternative numbering seeds.
        starts = _clockwise_nonfusion_block_starts(eligible, cycle)
        if not starts or any(start not in perimeter or len(perimeter[start]) != 2
                             for start in starts):
            return ()
        chosen_starts = _uppermost_nonfusion_block_starts(eligible, cycle, coords)
        if not chosen_starts:
            return ()
        for start in chosen_starts:
            for first in perimeter[start]:
                order = [start, first]
                while len(order) < len(perimeter):
                    options = perimeter[order[-1]] - {order[-2]}
                    if len(options) != 1:
                        break
                    order.append(next(iter(options)))
                if (len(order) == len(perimeter) and start in perimeter[order[-1]]
                        and set(order) == set(perimeter)):
                    area = _signed_area(tuple(coords[i] for i in order))
                    if area < -_EPS:
                        orders.append(_ClockwiseNumberingPath(
                            tuple(order),
                            tuple(tuple(sorted(faces[index])) for index in face_path),
                            tuple(transition_edges),
                            tuple(successor_edges)))
    return tuple(dict.fromkeys(orders))


def _clockwise_successor_face(graph, faces, mol, coords, face_index,
                              *, from_position=None):
    """Return the unique next face on the clockwise perimeter after a face.

    P-25.3.3.1.1 sends numbering to the next ring encountered clockwise when
    the selected uppermost ring has no nonfusion atom. The external perimeter
    gives an atom-map-independent traversal for that instruction. Multiple
    separated edge runs can imply distinct successor faces; those remain
    unresolved instead of being ordered by an arbitrary ray or graph index.
    """
    if not 0 <= face_index < len(faces):
        return None
    raw_cycle = tuple(getattr(graph, "perimeter_cycle", ()))
    perimeter_edges = frozenset(getattr(graph, "perimeter_edges", ()))
    if not perimeter_edges or any(atom not in coords for atom in raw_cycle):
        return None
    peripheral_atoms = frozenset().union(*perimeter_edges)
    if len(raw_cycle) != len(peripheral_atoms) or frozenset(raw_cycle) != peripheral_atoms:
        return None
    area = _signed_area(tuple(coords[atom] for atom in raw_cycle))
    if abs(area) <= _EPS:
        return None
    clockwise_cycle = raw_cycle if area < 0 else tuple(reversed(raw_cycle))

    edge_owners = {}
    for index, face in enumerate(faces):
        cycle = next(iter(_face_cycle_orders(mol, face)), None)
        if cycle is None:
            return None
        for left, right in zip(cycle, cycle[1:] + cycle[:1]):
            edge_owners.setdefault(frozenset((left, right)), []).append(index)
    edge_sequence = tuple(frozenset((left, right)) for left, right in
                          zip(clockwise_cycle, clockwise_cycle[1:] + clockwise_cycle[:1]))
    if set(edge_sequence) != set(perimeter_edges):
        return None
    owners = []
    for edge in edge_sequence:
        matches = edge_owners.get(edge, ())
        if len(matches) != 1:
            return None
        owners.append(matches[0])

    if from_position is None:
        run_starts = [position for position, owner in enumerate(owners)
                      if owner == face_index
                      and owners[(position - 1) % len(owners)] != face_index]
        # The selected ring itself must have one unambiguous external
        # perimeter run. Several runs do not define which ring is first.
        if len(run_starts) != 1:
            return None
        position = run_starts[0]
    else:
        if (isinstance(from_position, bool) or not isinstance(from_position, int)
                or not 0 <= from_position < len(owners)
                or owners[from_position] != face_index):
            return None
        position = from_position
    steps = 0
    while owners[(position + 1) % len(owners)] == face_index:
        position = (position + 1) % len(owners)
        steps += 1
        if steps >= len(owners):
            return None
    successor_position = (position + 1) % len(owners)
    successor_face = owners[successor_position]
    if successor_face == face_index:
        return None
    return successor_face, edge_sequence[successor_position], successor_position


def _signed_area(poly):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1]
                     for a, b in zip(poly, poly[1:] + poly[:1]))


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _vertex_collision(coords, atom, point):
    x, y = point
    close = math.isclose
    for other, (u, v) in coords.items():
        if other != atom and close(u, x, abs_tol=_EPS) and close(v, y, abs_tol=_EPS):
            return True
    return False


def _close_point(a, b):
    return math.isclose(a[0], b[0], abs_tol=_EPS) and math.isclose(a[1], b[1], abs_tol=_EPS)


def _connected(graph):
    if not graph:
        return False
    seen, stack = set(), [next(iter(graph))]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph[node] - seen)
    return len(seen) == len(graph)
