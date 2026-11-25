"""Amidine-bearing ring systems joined directly or by an independently owned linker."""

from dataclasses import dataclass
from itertools import product

from .amidine_parents import _suffix_decision
from .functional_parent_extensions import side
from .naming_methods import MethodDecision, _connector_name, _isolated_heterounits, _linear_order
from .parent_selection import locant_key, locant_set, parent_frontier
from .rooted_carbon_parents import hydride_numberings, skeleton_prefixes, substituent_numbering_key


@dataclass(frozen=True)
class RingEnd:
    ring: frozenset
    atoms: frozenset
    root: int
    connector: int


def _ring_end(mol, ring, all_rings):
    other_rings = all_rings - ring
    for root in ring:
        for bond in mol.GetAtomWithIdx(root).GetBonds():
            other = bond.GetOtherAtomIdx(root)
            if other in ring or bond.GetBondTypeAsDouble() != 1 or bond.GetStereo():
                continue
            atoms = side(mol, root, other)
            if (ring <= atoms and not atoms & other_rings and
                    sum(a.GetIdx() in atoms for a in mol.GetAtomWithIdx(other).GetNeighbors()) == 1):
                return RingEnd(ring, atoms, root, other)
    return None


def _direct_numberings(mol, Chem, ends):
    """P-28 assigns each ring junction before suffix and substituent locants."""
    rows = []
    for end in ends:
        choices = list(hydride_numberings(mol, Chem, end.root, end.ring))
        if not choices:
            return
        best = min((frame, locant_key(m[end.root])) for _, m, frame, _ in choices)
        rows.append([(word, m, m[end.root]) for word, m, frame, _ in choices
                     if (frame, locant_key(m[end.root])) == best])
    for first, second in (rows, rows[::-1]):
        for (left, a, p), (right, b, q) in product(first, second):
            if left != right or not left.isalpha():
                continue
            if left == 'benzene':
                word = 'biphenyl'
            elif left.startswith('cyclo'):
                word = 'bi(' + left + ')'  # Distinguish the assembly from a von Baeyer bicyclo parent.
            else:
                word = 'bi' + left
            yield (f"[{p},{q}'-{word}]", a | {i: loc + "'" for i, loc in b.items()},
                   (locant_key(p), locant_key(q)), ())


def _linker(mol, Chem, atoms, ends):
    ports = [e.connector for e in ends]
    if len(ends) == 2:
        path = _linear_order(mol, atoms, ports) if len(atoms) > 1 else ()
        if path:
            hetero = _isolated_heterounits(mol, path)
            if hetero is not None and len(hetero) >= 4:
                return ''  # P-51.4 skeletal replacement takes precedence.
    bridge = _connector_name(mol, atoms, ends, functional_units=True)
    if bridge or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 or mol.GetAtomWithIdx(i).IsInRing()
                     for i in atoms):
        return bridge
    # P-29. Include every free-valence site, including repeated sites such
    # as propane-2,2-diyl, and retain the original configuration of alkenes.
    from .component_mapping import MappedComponent, decorate_component
    from .formatting import attach_suffix
    options = []
    for word, mapping, frame, pi in hydride_numberings(mol, Chem, ports[0], atoms):
        if not all(i in mapping for i in ports):
            continue
        prefixes = skeleton_prefixes(mol, Chem, mapping, atoms)
        if prefixes is None:
            continue
        positions = sorted((mapping[i] for i in ports), key=locant_key)
        parent = attach_suffix(word, {2: 'diyl', 3: 'triyl', 4: 'tetrayl'}[len(ends)], positions)
        key = substituent_numbering_key(mol, mapping, prefixes)
        component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping),
                                    tuple(prefixes), atoms - mapping.keys(), systematic_parent=True)
        name = decorate_component(mol, Chem, component)
        if name and key is not None:
            options.append(((frame, locant_set(positions), pi, key), name))
    if not options:
        return ''
    best = min(k for k, _ in options)
    winners = {name for k, name in options if k == best}
    return winners.pop() if len(winners) == 1 else ''


def linked_imidic_parent(mol, Chem, hosts, sites, n_prefixes, arms, scope):
    from .multiplicative_parent_extensions import _assembly_name

    entire_core = set.union(*(s.atoms for s in sites))
    all_rings = frozenset.union(*(frozenset(r) for r, _ in hosts))
    ends = [_ring_end(mol, ring, all_rings) for ring, _ in hosts]
    if all(ends) and len(ends) == 2 and ends[0].connector == ends[1].root and ends[1].connector == ends[0].root:
        direct = _suffix_decision(mol, Chem, sites, [], entire_core, n_prefixes, arms, scope,
                                  numberings=_direct_numberings(mol, Chem, ends))
        if direct:
            return direct
    if 2 <= len(ends) <= 4 and all(ends):
        residues = frozenset.union(*(e.atoms for e in ends))
        connector = scope - residues
        if (sum(len(e.atoms) for e in ends) == len(residues) and
                all(e.connector in connector for e in ends) and not any(
                mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetChiralTag() for i in connector)):
            bridge = _linker(mol, Chem, connector, ends)
            if bridge:
                units = []
                for end, (_, group) in zip(ends, hosts):
                    core = set.union(*(s.atoms for s in group))
                    branches = [b for b in n_prefixes if b[0] in core]
                    if not core <= end.atoms or any(not b[2] <= end.atoms for b in branches):
                        break
                    unit = _suffix_decision(mol, Chem, group, [], core, branches, arms & end.atoms,
                                            end.atoms, attachment=end.root)
                    if unit is None:
                        break
                    pos = dict(unit.atom_locants)[end.root]
                    units.append((unit.name, pos, bool(end.atoms - end.ring - core)))
                if len(units) == len(ends) and len(set(units)) == 1:
                    name, loc, substituted = units[0]
                    name = _assembly_name(name, loc, bridge, len(units), substituted=substituted,
                                          simple_bridge=len(connector) == 1)
                    return MethodDecision(name, 'multiplicative', frozenset(s.center for s in sites),
                                          scope, unit.rule + '/P-51.3.1')
    # For inequivalent ends compare single-parent constructions, keeping the
    # other amidines in fully owned, independently rendered substituents.
    candidates = []
    most = max(len(group) for _, group in hosts)
    for ring, group in hosts:
        if len(group) != most:
            continue
        core = set.union(*(s.atoms for s in group))
        branches = [b for b in n_prefixes if b[0] in core]
        candidate = _suffix_decision(mol, Chem, group, [], core, branches,
                                    set().union(*(b[2] for b in branches)), scope, parent_facts=ring)
        if candidate is None:
            return None
        candidates.append(candidate)
    frontier = parent_frontier(candidates, lambda row: row[1])
    if frontier and len({d.name for d, _ in frontier}) == 1:
        return frontier[0][0]
    return None
