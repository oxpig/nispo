"""Graph proof for a bounded linked saturated spiro/bridged parent context.

This only identifies a P-24.2.1 monospiro core whose two components are
monocyclic. It does not decide that core's seniority over the other ring
system, select a name, or create a preferred-name claim.
"""
from __future__ import annotations

from .constants import ROOTS
from .ring_topology import ring_system_topologies


def _components(graph, removed):
    remaining = set(graph) - set(removed)
    components = []
    while remaining:
        start = remaining.pop()
        group, pending = {start}, [start]
        while pending:
            for other in graph[pending.pop()] & remaining:
                remaining.remove(other)
                group.add(other)
                pending.append(other)
        components.append(frozenset(group))
    return tuple(components)


def _spiro_core(topology):
    graph = topology.graph
    if topology.cycle_rank != 2 or len(topology.atoms) not in ROOTS:
        return False
    centers = [idx for idx, neighbors in graph.items() if len(neighbors) == 4]
    if len(centers) != 1 or any(len(neighbors) != 2 for idx, neighbors in graph.items()
                                if idx != centers[0]):
        return False
    center = centers[0]
    parts = _components(graph, {center})
    return (len(parts) == 2 and all(
        len(part) >= 2 and len(graph[center] & part) == 2
        and sum(len(graph[i] & part) for i in part) == 2 * (len(part) - 1)
        for part in parts))


def _bridged_theta(topology):
    graph = topology.graph
    if topology.cycle_rank != 2 or len(topology.atoms) not in ROOTS:
        return False
    heads = tuple(idx for idx, neighbors in graph.items() if len(neighbors) == 3)
    if (len(heads) != 2 or heads[1] in graph[heads[0]]
            or any(len(neighbors) != 2 for idx, neighbors in graph.items()
                   if idx not in heads)):
        return False
    parts = _components(graph, heads)
    return (len(parts) == 3 and all(
        len(graph[heads[0]] & part) == 1 and len(graph[heads[1]] & part) == 1
        and sum(len(graph[i] & part) for i in part) == 2 * (len(part) - 1)
        for part in parts))


def linked_saturated_spiro_bicyclic_core(mol, Chem):
    """Return only the exact spiro atom scope, or an empty set if unproved.

    Scope: one saturated C/C spiro ring system, one separate saturated C/C
    nonfused theta system, and one or two methylene atoms linking them. The
    complete neutral all-carbon graph, its ring blocks, linker, and lack of
    annotations must be established before excluding a spiro component
    template. Other parent candidates and any preferred comparison remain
    untouched. Ring topology uses the actual graph, not SSSR ring count.
    """
    if mol.GetRingInfo().NumRings() < 4 or mol.GetStereoGroups():
        return frozenset()
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() != 6 or a.GetIsotope() or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery()
           for a in atoms):
        return frozenset()
    if any(b.GetBondTypeAsDouble() != 1 or b.GetIsAromatic() or b.HasQuery()
           or int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()):
        return frozenset()
    tops = ring_system_topologies(mol, Chem)
    if len(tops) != 2:
        return frozenset()
    spi = [top for top in tops if _spiro_core(top)]
    bridged = [top for top in tops if _bridged_theta(top)]
    if len(spi) != 1 or len(bridged) != 1 or spi[0].atoms & bridged[0].atoms:
        return frozenset()
    spiro, other = spi[0].atoms, bridged[0].atoms
    linker = frozenset(a.GetIdx() for a in atoms if a.GetIdx() not in spiro | other)
    if not 1 <= len(linker) <= 2 or any(mol.GetAtomWithIdx(i).GetDegree() != 2
                                         or mol.GetAtomWithIdx(i).GetTotalNumHs() != 2
                                         for i in linker):
        return frozenset()
    cuts = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()) for i in linker}
    core_links = [(i, n) for i, neighbors in cuts.items() for n in neighbors if n in spiro]
    other_links = [(i, n) for i, neighbors in cuts.items() for n in neighbors if n in other]
    if len(core_links) != 1 or len(other_links) != 1:
        return frozenset()
    if any(n not in linker | spiro | other for neighbors in cuts.values() for n in neighbors):
        return frozenset()
    if len(linker) == 2:
        left, right = tuple(linker)
        if right not in cuts[left]:
            return frozenset()
    if any(n.GetIdx() in other for i in spiro for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return frozenset()
    return spiro


def order_linked_spiro_scaffold_facts(mol, Chem, facts):
    """Order a complete, single-template spiro numbering set by P-45.2.2.

    The graph proof admits exactly one substituent attachment to the spiro
    core. Require the entire automorphism set for the numeric monospiro
    parent: four numberings for unlike rings, eight for identical rings.
    Any competing template or incomplete numbering set is returned unchanged.
    """
    core = linked_saturated_spiro_bicyclic_core(mol, Chem)
    if not core or not facts:
        return facts
    topology = next((top for top in ring_system_topologies(mol, Chem)
                     if top.atoms == core), None)
    if topology is None:
        return facts
    center = next((idx for idx, neighbors in topology.graph.items() if len(neighbors) == 4), None)
    if center is None:
        return facts
    parts = _components(topology.graph, {center})
    sizes = tuple(sorted(len(part) for part in parts))
    if len(sizes) != 2:
        return facts
    expected_count = 8 if sizes[0] == sizes[1] else 4
    expected_parent = f"spiro[{sizes[0]}.{sizes[1]}]{ROOTS[len(core)]}ane"
    if len(facts) != expected_count:
        return facts
    boundary = [(i, n.GetIdx()) for i in core for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() not in core]
    if len(boundary) != 1:
        return facts
    host = boundary[0][0]
    from .template_access import template_record_locants
    scored = []
    for fact in facts:
        if (frozenset(fact.match) != core or len(fact.match) != len(core)
                or fact.compiled.record.name != expected_parent):
            return facts
        locants = template_record_locants(fact.template)
        if set(locants.values()) != {str(i) for i in range(1, len(core) + 1)}:
            return facts
        locant = locants.get(fact.match.index(host), '')
        if not str(locant).isdigit():
            return facts
        scored.append((int(locant), fact))
    if len({tuple(f.match) for f in facts}) != expected_count:
        return facts
    return tuple(fact for _locant, fact in sorted(scored, key=lambda item: item[0]))
