"""Mapped, bounded P-63.6 names for neutral organic chalcogen oxides.

The producer covers carbon-only path sides (including internal attachment
sites) and simple carbocyclic monocyclic sides attached to one neutral
S/Se/Te center with one or two terminal oxo atoms. It returns source-backed substitutive names, and the
published identical-ring multiplicative form where its strict conditions are
met.  Unsupported functions and annotations decline rather than fall through
to an untyped sulfur spelling.
"""

from dataclasses import dataclass, replace

from .molecule_iteration import iter_atoms, iter_bonds
from .branch_rules import parent_chain_name
from .component_mapping import MappedComponent, decorate_component
from .formatting import complete_parent_name
from .naming_methods import MethodDecision
from .parent_selection import graph_parent_facts, parent_frontier, compare_parents


_OXIDE_PREFIX = {
    (16, 1): ("sulfinyl", "sulfinyl"),
    (16, 2): ("sulfonyl", "sulfonyl"),
    (34, 1): ("seleninyl", "seleninyl"),
    (34, 2): ("selenonyl", "selenonyl"),
    (52, 1): ("tellurinyl", "tellurinyl"),
    (52, 2): ("telluronyl", "telluronyl"),
}


@dataclass(frozen=True, slots=True)
class _Oxide:
    center: int
    element: int
    oxo: tuple[int, ...]
    sides: tuple[frozenset[int], frozenset[int]]
    roots: tuple[int, int]
    prefixes: tuple[str, str]
    connector: str


@dataclass(frozen=True, slots=True)
class _Parent:
    atoms: frozenset[int]
    locants: tuple[tuple[int, str], ...]
    parent: str
    kind: str
    site: str


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _recognize(mol):
    centers = [a for a in iter_atoms(mol) if a.GetAtomicNum() in {16, 34, 52}]
    if len(centers) != 1:
        return None
    center = centers[0]
    eidx, element = center.GetIdx(), center.GetAtomicNum()
    if (center.GetFormalCharge() or center.GetNumRadicalElectrons()
            or int(center.GetChiralTag()) or center.IsInRing()):
        return None
    oxo = tuple(sorted(n.GetIdx() for n in center.GetNeighbors()
                       if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                       and not n.GetFormalCharge()
                       and mol.GetBondBetweenAtoms(eidx, n.GetIdx()).GetBondTypeAsDouble() == 2.0))
    if len(oxo) not in {1, 2}:
        return None
    # P-63.6 sulfinyl/sulfonyl groups have no hydrogens on the central
    # chalcogen. A hydrogen-rich lambda hydride is another structure, even
    # when its heavy-atom neighbours happen to resemble a sulfoxide.
    if (center.GetTotalNumHs(includeNeighbors=True)
            or center.GetTotalValence() != 2 + 2 * len(oxo)):
        return None
    if len({mol.GetAtomWithIdx(i).GetIsotope() for i in oxo}) != 1:
        return None
    sides = [n for n in center.GetNeighbors() if n.GetAtomicNum() == 6
             and _single(mol, eidx, n.GetIdx())]
    if len(sides) != 2 or len(sides) + len(oxo) != center.GetDegree():
        return None
    components = []
    for root_atom in sides:
        root = root_atom.GetIdx()
        found, stack = {root}, [root]
        while stack:
            idx = stack.pop()
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other == eidx:
                    continue
                bond = mol.GetBondBetweenAtoms(idx, other)
                if (neighbor.GetAtomicNum() != 6 or bond is None
                        or bond.GetBondTypeAsDouble() not in {1.0, 1.5}):
                    return None
                if other not in found:
                    found.add(other); stack.append(other)
        components.append((root, frozenset(found)))
    if components[0][1] & components[1][1]:
        return None
    # Every heavy atom must be in one carbon side, the central atom, or a
    # terminal oxo atom. This excludes unproved groups and hidden branches.
    side_atoms = components[0][1] | components[1][1]
    expected = side_atoms | {eidx, *oxo}
    if expected != {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}:
        return None
    for atom in iter_atoms(mol):
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        if number == 1:
            if atom.GetIsotope():
                return None
        elif atom.GetIsotope() and idx not in side_atoms | {eidx, *oxo}:
            return None
        if int(atom.GetChiralTag()) and (number != 6 or idx not in side_atoms):
            return None
    if mol.GetStereoGroups():
        return None
    words = _OXIDE_PREFIX.get((element, len(oxo)))
    if not words:
        return None
    return _Oxide(eidx, element, oxo,
                  (components[0][1], components[1][1]),
                  (components[0][0], components[1][0]),
                  (words[0], words[0]), words[1])


def _chain_parent(mol, atoms, root):
    if not atoms or len(atoms) > 12 or any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return ()
    graph = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms)
             for i in atoms}
    if len(atoms) == 1:
        paths = ((root,),)
    else:
        if sum(map(len, graph.values())) != 2 * (len(atoms) - 1):
            return ()
        ends = sorted(i for i, ns in graph.items() if len(ns) == 1)
        if len(ends) != 2:
            return ()
        paths = []
        for start in ends:
            path, prev, here = [], -1, start
            while True:
                path.append(here)
                choices = [n for n in graph[here] if n != prev]
                if not choices:
                    break
                if len(choices) != 1:
                    return ()
                prev, here = here, choices[0]
            if len(path) == len(atoms) and root in path:
                paths.append(tuple(path))
        paths = tuple(paths)
    output = []
    for path in paths:
        base = parent_chain_name(mol, path)
        if not base:
            continue
        locants = tuple((idx, str(i + 1)) for i, idx in enumerate(path))
        output.append(_Parent(frozenset(path), locants, complete_parent_name(base), "chain",
                              dict(locants)[root]))
    return tuple(output)


def _ring_parents(mol, atoms, root):
    if not atoms or len(atoms) > 10 or any(not mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return ()
    bonds = [b for b in iter_bonds(mol) if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    if len(bonds) != len(atoms):
        return ()
    graph = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms)
             for i in atoms}
    if any(len(ns) != 2 for ns in graph.values()):
        return ()
    from .constants import ROOTS
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
    if aromatic and len(atoms) != 6:
        return ()
    if not aromatic and any(b.GetBondTypeAsDouble() != 1 for b in bonds):
        return ()
    size = len(atoms)
    if size not in ROOTS or not 3 <= size <= 10:
        return ()
    parent = "benzene" if aromatic else "cyclo" + ROOTS[size] + "ane"
    result = []
    for first in sorted(graph[root]):
        order, prev, here = [root], root, first
        while here != root and here not in order:
            order.append(here)
            nxt = [x for x in graph[here] if x != prev]
            if len(nxt) != 1:
                break
            prev, here = here, nxt[0]
        if here != root or len(order) != size:
            continue
        mapping = tuple((idx, str(i + 1)) for i, idx in enumerate(order))
        record = _Parent(frozenset(order), mapping, parent, "ring", "1")
        if record not in result:
            result.append(record)
    return tuple(result)


def _side_parents(mol, side, root):
    atoms = frozenset(side)
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
        return ()
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return _ring_parents(mol, atoms, root)
    return _chain_parent(mol, atoms, root)


def _side_is_unannotated(mol, atoms):
    return all(not mol.GetAtomWithIdx(i).GetIsotope()
               and not int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms)


def _requires_isotope_locants(mol, atoms):
    """Whether carbon labels require positional locants on this parent.

    P-82.6.1.1 requires every normally omitted locant once an isotope
    descriptor needs positional locants; P-82.6.1.2/.3 retain the exceptions
    for one atom of the element and uniform modification of every position.
    """
    isotopes = [mol.GetAtomWithIdx(i).GetIsotope() for i in atoms]
    labeled = [mass for mass in isotopes if mass]
    if not labeled or len(atoms) == 1:
        return False
    return not (len(labeled) == len(atoms) and len(set(labeled)) == 1)


def _alternative_facts(mol, parent):
    facts = graph_parent_facts(mol, parent.atoms, dict(parent.locants),
        principal_class=(40, 0), prefixes=((parent.site, "oxide-linker"),),
        kind=parent.kind, name=parent.parent, allow_annotations=True)
    return (replace(facts, substituent_nonstandard_key=(),
                    substituent_isotope_key=()) if facts is not None else None)


def _decision_for_parent(mol, Chem, oxide, parent_side, other_side, acyl_prefix):
    root = oxide.roots[oxide.sides.index(parent_side)]
    remaining = frozenset((set(oxide.sides[0]) | set(oxide.sides[1]) |
                           {oxide.center, *oxide.oxo}) - set(parent_side))
    built = []
    for parent in _side_parents(mol, parent_side, root):
        site = parent.site
        # A single substituent on a simple ring has a unique locant and takes
        # no locant. P-14.3.4.2 also omits locant 1 for a monosubstituted
        # homogeneous two-carbon chain. If a partial isotope label needs a
        # locant, P-82.6.1.1 requires the normally omitted locant as well.
        omit_prefix_locant = (parent.kind == "ring" or len(parent.atoms) == 1
                              or len(parent.atoms) == 2)
        if _requires_isotope_locants(mol, parent.atoms):
            omit_prefix_locant = False
        prefix_locant = "" if omit_prefix_locant else site
        # P-45 comparison needs the actual attachment locant even when the
        # printed locant is omitted by the single-position rule.
        fact_prefixes = ((site, acyl_prefix),)
        # P-63.6 presents these acyl groups as complete bracketed units (for
        # example, (ethanesulfinyl)butane); keep the enclosure explicit.
        full_parent = (f"{prefix_locant}-({acyl_prefix}){parent.parent}"
                       if prefix_locant else f"({acyl_prefix}){parent.parent}")
        mapped = MappedComponent(parent=full_parent, atom_locants=parent.locants,
            atoms=parent.atoms, embedded_atoms=remaining,
            systematic_parent=True)
        name = decorate_component(mol, Chem, mapped)
        if not name:
            continue
        facts = graph_parent_facts(mol, parent.atoms, dict(parent.locants),
            principal_class=(40, 0), prefixes=fact_prefixes, kind=parent.kind,
            name=name, allow_annotations=True)
        if facts is not None:
            facts = replace(facts, substituent_nonstandard_key=(),
                            substituent_isotope_key=())
            built.append((name, facts, parent, mapped))
    return built


def _complete_roles(mol, oxide, parent, parent_component, acyl_parent, method):
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    core = frozenset({oxide.center, *oxide.oxo})
    acyl_locants = acyl_parent.locants + tuple((i, "") for i in sorted(core))
    roles = [("parent-hydride", parent.atoms, parent.locants),
             ("chalcogen-oxide-prefix", acyl_parent.atoms | core, acyl_locants)]
    flat = [idx for _, atoms, _ in roles for idx in atoms]
    if len(flat) != len(set(flat)) or frozenset(flat) != all_heavy:
        return None
    return MethodDecision(name=method[0], method=method[1],
        principal_centers=frozenset({oxide.center}), atoms=all_heavy,
        rule="P-63.6/P-65.3.2.2.2/P-51.1/P-44/P-45",
        atom_locants=parent_component.atom_locants + acyl_locants,
        component_roles=tuple(roles))


def _acyl_group_name(parent, suffix):
    """Form a source-derived acyl group from its mapped acid-side hydride."""
    if parent.kind != "chain" or len(parent.atoms) <= 2:
        return parent.parent + suffix
    # P-65.3.2.2.2 replaces only the `ic acid` ending with `yl`, preserving
    # the full hydride and its suffix locant. Unlike the terminal `oic acid`
    # classes in P-14.3.4.1, sulfur/selenium/tellurium acids have no terminal
    # locant omission; for a chain of three or more atoms both 1 and internal
    # positions are explicit.
    return f"{parent.parent}-{parent.site}-{suffix}"


def _same_simple_cyclic_sides(mol, oxide):
    left, right = oxide.sides
    if not _side_is_unannotated(mol, left) or not _side_is_unannotated(mol, right):
        return False
    if len(left) != len(right):
        return False
    def signature(atoms):
        return (len(atoms), all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms),
                tuple(sorted(mol.GetBondBetweenAtoms(b.GetBeginAtomIdx(), b.GetEndAtomIdx()).GetBondTypeAsDouble()
                             for b in iter_bonds(mol) if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms)))
    return signature(left) == signature(right) and all(mol.GetAtomWithIdx(i).IsInRing()
                                                       for i in left | right)


def _multiplicative_cyclic(mol, Chem, oxide):
    if not _same_simple_cyclic_sides(mol, oxide):
        return None
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic()
                   for i in oxide.sides[0] | oxide.sides[1])
    if aromatic and len(oxide.sides[0]) != 6:
        return None
    left, right = oxide.sides
    left_root, right_root = oxide.roots
    def ring_map(atoms, root, prime):
        graph = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms)
                 for i in atoms}
        order = [root]
        prev, here = root, min(graph[root])
        while here != root and here not in order:
            order.append(here)
            nxt = [i for i in graph[here] if i != prev]
            if len(nxt) != 1: return None
            prev, here = here, nxt[0]
        if here != root or len(order) != len(atoms): return None
        return tuple((idx, f"{i + 1}{prime}") for i, idx in enumerate(order))
    maps = ring_map(left, left_root, "")
    maps2 = ring_map(right, right_root, "′")
    if maps is None or maps2 is None:
        return None
    linker = oxide.connector
    from .constants import ROOTS
    parent = "benzene" if aromatic else "cyclo" + ROOTS[len(left)] + "ane"
    name = f"1,1′-{linker}di{parent}"
    atom_map = maps + maps2 + ((oxide.center, ""),) + tuple((i, "") for i in oxide.oxo)
    comp_atoms = left | right
    embedded = frozenset({oxide.center, *oxide.oxo})
    # The multiplicative word itself contains and owns all mapped parent and
    # linker atoms; central oxidation atoms have no separately rendered label.
    # decorate_component's component partition is therefore represented by
    # the explicit complete map rather than an embedded atom overlap.
    complete = MappedComponent(parent=name, atom_locants=atom_map,
        atoms=comp_atoms | embedded, systematic_parent=True)
    rendered = decorate_component(mol, Chem, complete)
    if not rendered:
        return None
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    roles = (("left-ring", left, maps), ("right-ring", right, maps2),
             ("chalcogen-linker", embedded, tuple((i, "") for i in sorted(embedded))))
    owned = [i for _, atoms, _ in roles for i in atoms]
    if len(owned) != len(set(owned)) or frozenset(owned) != all_heavy:
        return None
    return MethodDecision(name=rendered, method="multiplicative",
        principal_centers=frozenset({oxide.center}), atoms=all_heavy,
        rule="P-63.6(3)/P-51.1.5/P-51.3.1/P-44",
        atom_locants=atom_map, component_roles=roles)


def chalcogen_oxide_decision(mol, Chem):
    """Build a complete mapped P-63.6 decision, or return ``None``.

    A central chalcogen isotope and uniform oxo isotope population have a
    separate descriptor on the complete oxide prefix (P-82.2.1/.5). Carbon
    isotope/stereo annotations are accepted only on the selected simple parent
    side, where the existing component decorator retains original atom maps.
    """
    if not mol.GetNumAtoms() or mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups():
        return None
    oxide = _recognize(mol)
    if oxide is None:
        return None
    core = frozenset({oxide.center, *oxide.oxo})
    core_labels = any(mol.GetAtomWithIdx(i).GetIsotope() for i in core)
    if any(a.GetIsotope() and Chem.GetPeriodicTable().GetMassForIsotope(
            a.GetAtomicNum(), a.GetIsotope()) <= 0
           for a in (mol.GetAtomWithIdx(i) for i in core)):
        return None
    if not core_labels and _same_simple_cyclic_sides(mol, oxide):
        multiplicative = _multiplicative_cyclic(mol, Chem, oxide)
        if multiplicative is not None:
            return multiplicative
    candidates = []
    for index in (0, 1):
        parent_side, other_side = oxide.sides[index], oxide.sides[1-index]
        if not _side_is_unannotated(mol, other_side):
            continue
        roots = oxide.roots[1-index]
        ligand_parents = _side_parents(mol, other_side, roots)
        if ligand_parents and ligand_parents[0].kind == "chain":
            # This is an acyl group derived from the corresponding oxoacid;
            # its suffix-bearing attachment locant is numbered low before
            # conversion to the P-65.3.2.2.2 `yl` form.
            lowest = min(int(parent.site) for parent in ligand_parents)
            ligand_parents = tuple(parent for parent in ligand_parents
                                   if int(parent.site) == lowest)
        ligand_rows = set()
        for p in ligand_parents:
            text = _acyl_group_name(p, oxide.prefixes[1-index])
            if core_labels:
                component = MappedComponent(
                    text, p.locants + tuple((i, '') for i in sorted(core)),
                    p.atoms | core, systematic_parent=True)
                text = decorate_component(mol, Chem, component)
            if text:
                ligand_rows.add((p, text))
        ligand_names = {name for _, name in ligand_rows}
        if len(ligand_names) != 1 or not ligand_rows:
            continue
        rows = _decision_for_parent(mol, Chem, oxide, parent_side, other_side,
                                    next(iter(ligand_names)))
        if not _side_is_unannotated(mol, parent_side) and rows:
            # Isotope/stereo grammar is available only when the annotated
            # side remains the selected parent. Prove that the competing
            # unannotated side is inferior on a decisive P-44 criterion;
            # otherwise decline instead of silently choosing a convenient
            # annotation placement.
            other_parents = _side_parents(mol, other_side, roots)
            competitors = [_alternative_facts(mol, p) for p in other_parents]
            if not competitors or any(f is None for f in competitors):
                continue
            if any(compare_parents(row[1], competitor).preference != -1
                   or not compare_parents(row[1], competitor).rule.startswith("P-44")
                   for row in rows for competitor in competitors):
                continue
        acyl_parent = next(iter(ligand_rows))[0]
        candidates.extend((row, index, parent_side, other_side, acyl_parent)
                          for row in rows)
    if not candidates:
        return None
    frontier = parent_frontier(candidates, lambda item: item[0][1])
    if not frontier:
        return None
    names = {row[0][0] for row in frontier}
    if len(names) != 1:
        return None
    for i, left in enumerate(frontier):
        for right in frontier[i+1:]:
            if compare_parents(left[0][1], right[0][1]).preference not in {-1, 0}:
                return None
    row, index, parent_side, other_side, acyl_parent = frontier[0]
    name, facts, parent, mapped = row
    decision = _complete_roles(mol, oxide, parent, mapped, acyl_parent,
                              (name, "substitutive"))
    return decision
