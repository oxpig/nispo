"""Mapped one-methano bridges on an admitted all-carbon fused hydride.

The caller supplies a fused parent template and its original-atom mapping.
The low-level writer accepts a supplied decomposition. The public selector
below discovers retained fused cores and compares the admitted decompositions;
broader unimplemented bridge alternatives remain explicit preference limits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


RULE = "P-25.4.1.1/P-25.4.1.2(a)/P-25.4.1.4/P-25.4.1.6/P-25.4.2.1.1/P-25.4.3.2.1/P-25.4.3.4/P-25.4.3.4.1/P-25.4.4/P-25.7.1.3.3"
_HYDROGEN = re.compile(r"(?<![A-Za-z0-9])((?:\d+H,)*\d+H)-")


def _locant_key(locant: str):
    match = re.fullmatch(r"(\d+)([a-z]?)", str(locant))
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


@dataclass(frozen=True)
class FusedCoreProof:
    """Caller attestation of constituent rings in an admitted fused template.

    ``rings`` gives each constituent ring as a cyclic sequence of template atom
    indices.  The producer validates the graph and endpoint membership; the
    caller remains responsible for selecting a source-supported fused parent.
    """

    template_name: str
    rings: tuple[tuple[int, ...], ...]


def _proof_is_consistent(query, template_name, proof):
    if (not isinstance(proof, FusedCoreProof) or proof.template_name != template_name
            or len(proof.rings) < 2):
        return False
    ring_edges = set()
    ring_atoms = set()
    normalized = []
    for ring in proof.rings:
        if (not isinstance(ring, tuple) or len(ring) < 3
                or any(isinstance(i, bool) or not isinstance(i, int)
                       or i < 0 or i >= query.GetNumAtoms() for i in ring)
                or len(set(ring)) != len(ring)):
            return False
        edges = set()
        for left, right in zip(ring, ring[1:] + ring[:1]):
            bond = query.GetBondBetweenAtoms(left, right)
            if bond is None:
                return False
            edges.add(frozenset((left, right)))
        normalized.append((set(ring), edges))
        ring_atoms.update(ring)
        ring_edges.update(edges)
    graph_edges = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                   for bond in query.GetBonds()}
    if ring_atoms != set(range(query.GetNumAtoms())) or ring_edges != graph_edges:
        return False
    # A fused system has constituent rings sharing an edge. Reject spiro-only
    # or disjoint ring proofs even when RDKit reports multiple rings.
    if not any(len(left_atoms & right_atoms) == 2
               and len(left_edges & right_edges) == 1
               for i, (left_atoms, left_edges) in enumerate(normalized)
               for right_atoms, right_edges in normalized[i + 1:]):
        return False
    # An articulation atom marks a spiro/disconnected-core decomposition.
    for removed in range(query.GetNumAtoms()):
        remaining = set(range(query.GetNumAtoms())) - {removed}
        if not remaining:
            return False
        seen = set()
        stack = [next(iter(remaining))]
        while stack:
            atom_id = stack.pop()
            if atom_id in seen:
                continue
            seen.add(atom_id)
            for neighbor in query.GetAtomWithIdx(atom_id).GetNeighbors():
                nxt = neighbor.GetIdx()
                if nxt in remaining and nxt not in seen:
                    stack.append(nxt)
        if seen != remaining:
            return False
    return True


def _next_bridge_locant(locants):
    keys = [_locant_key(str(locant)) for locant in locants.values()]
    if not keys or any(key is None for key in keys):
        return None
    return str(max(keys)[0] + 1)


def _without_indicated_hydrogen(name: str):
    match = _HYDROGEN.search(name)
    if match is None:
        return "", name
    indicated = match.group(1)
    return indicated, name[:match.start()] + name[match.end():]


def _same_core_graph(mol, query, match):
    """Require the mapped core-edge set to equal the numbered template graph."""
    mapped = set(match)
    expected = set()
    for bond in query.GetBonds():
        left, right = match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()]
        actual = mol.GetBondBetweenAtoms(left, right)
        if actual is None:
            return False
        if bond.GetIsAromatic():
            if not actual.GetIsAromatic():
                return False
        elif actual.GetBondTypeAsDouble() != bond.GetBondTypeAsDouble():
            return False
        expected.add(frozenset((left, right)))
    actual_edges = set()
    for atom_id in mapped:
        atom = mol.GetAtomWithIdx(atom_id)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other in mapped:
                actual_edges.add(frozenset((atom_id, other)))
    return actual_edges == expected


def _core_name_on_original_graph(mol, template, match):
    """Use the existing fused-hydride state mapper against original atoms."""
    from .preferred_fused import hydrogenated_fused_parent_name

    name = hydrogenated_fused_parent_name(mol, template, match)
    if name:
        return name
    # The helper returns an empty string for the unchanged template state. Only
    # accept that fallback if every original core atom has the same hydrogen
    # count after accounting for the new bridge bond as an external bond.
    query = template["mol"]
    core = set(match)
    for j, atom_id in enumerate(match):
        atom = mol.GetAtomWithIdx(atom_id)
        ref = query.GetAtomWithIdx(j)
        external = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()
                       if b.GetOtherAtomIdx(atom_id) not in core)
        if atom.GetTotalNumHs() + external - ref.GetTotalNumHs() != 0:
            return None
    name = str(template["name"])
    if _HYDROGEN.search(name) is None:
        # Preserve indicated hydrogen on the original mapped fused core even
        # when no additional hydro prefix is needed. The cited locant follows
        # P-25.7.1.3.3 and is moved to the front of the complete bridge name.
        indicated = []
        for j, atom_id in enumerate(match):
            ref = query.GetAtomWithIdx(j)
            atom = mol.GetAtomWithIdx(atom_id)
            if (ref.GetAtomicNum() == 6 and not ref.GetIsAromatic()
                    and ref.GetTotalNumHs() == 2
                    and all(b.GetBondTypeAsDouble() == 1.0 for b in ref.GetBonds())
                    and atom.GetTotalNumHs() > 0
                    and all(mol.GetBondBetweenAtoms(atom_id, n.GetIdx()).GetBondTypeAsDouble() == 1.0
                            for n in atom.GetNeighbors() if n.GetIdx() in core)):
                locant = str(template["locants"].get(j, ""))
                if not _locant_key(locant):
                    return None
                indicated.append(locant + "H")
        if indicated:
            indicated.sort(key=lambda x: _locant_key(x[:-1]))
            name = ",".join(indicated) + "-" + name
    return name


def methano_fused_decision(mol, Chem, template, match, *, core_proof=None):
    """Return a complete mapped name for one CH2 bridge on a supplied core.

    Scope is exactly one numbered all-carbon fused parent core plus one neutral
    CH2 bridge atom. The bridge has two single bonds to distinct core atoms and
    accounts, with the core mapping, for all heavy atoms. ``template`` must
    carry the source-supported core name, molecule, and locants; ``match`` maps
    template atom order to original molecule atom IDs. ``core_proof`` is a
    ``FusedCoreProof`` supplied by the caller with constituent ring cycles;
    this helper validates those cycles and implements P-25.4.1.2(a) only.
    """
    if (mol.GetNumHeavyAtoms() > 96 or mol.GetStereoGroups()
            or not isinstance(template, dict) or not isinstance(match, (tuple, list))):
        return None
    query = template.get("mol")
    name = template.get("name")
    locants = template.get("locants")
    if (query is None or not isinstance(name, str) or not name.endswith("e")
            or not isinstance(locants, dict) or len(match) != query.GetNumAtoms()
            or query.GetRingInfo().NumRings() < 2 or not query.GetNumAtoms()
            or not _proof_is_consistent(query, name, core_proof)):
        return None
    if (any(a.GetAtomicNum() != 6 or a.GetFormalCharge() or a.GetIsotope()
            or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or not a.IsInRing()
            for a in query.GetAtoms())
            or set(locants) != set(range(query.GetNumAtoms()))):
        return None
    if (any(isinstance(i, bool) or not isinstance(i, int)
            or i < 0 or i >= mol.GetNumAtoms() for i in match)
            or len(set(match)) != len(match)):
        return None
    core = frozenset(match)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    extras = heavy - core
    if len(extras) != 1:
        return None
    bridge_id = next(iter(extras))
    bridge = mol.GetAtomWithIdx(bridge_id)
    if (bridge.GetAtomicNum() != 6 or bridge.GetFormalCharge() or bridge.GetIsotope()
            or bridge.GetNumRadicalElectrons() or int(bridge.GetChiralTag())
            or bridge.GetTotalNumHs() != 2 or bridge.IsInRing() is False):
        return None
    bridge_neighbors = [n.GetIdx() for n in bridge.GetNeighbors() if n.GetAtomicNum() > 1]
    if (len(bridge_neighbors) != 2 or len(set(bridge_neighbors)) != 2
            or not set(bridge_neighbors) <= core):
        return None
    if any(mol.GetBondBetweenAtoms(bridge_id, i).GetBondTypeAsDouble() != 1.0
           for i in bridge_neighbors):
        return None
    for atom in mol.GetAtoms():
        if (atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope() or int(atom.GetChiralTag())):
            return None
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                return None
    # This bounded operation handles only unannotated bond topology. In
    # particular, directional and E/Z bond annotations must not be discarded.
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE
           or bond.GetBondDir() != Chem.BondDir.NONE for bond in mol.GetBonds()):
        return None
    if not _same_core_graph(mol, query, match):
        return None
    if any(a.GetIdx() not in core | {bridge_id} for a in mol.GetAtoms()
           if a.GetAtomicNum() > 1):
        return None

    mol_to_template = {mol_id: template_id for template_id, mol_id in enumerate(match)}
    bridgehead_template_ids = [mol_to_template[i] for i in bridge_neighbors]
    # Bounded to P-25.4.1.2(a): endpoints must be nonadjacent positions on
    # one explicitly attested constituent ring. Criterion (b) is intentionally
    # left for a separately proved route.
    endpoint_pair = set(bridgehead_template_ids)
    if not any(endpoint_pair <= set(ring)
               and query.GetBondBetweenAtoms(*bridgehead_template_ids) is None
               for ring in core_proof.rings):
        return None
    bridgehead_locants = [str(locants[i]) for i in bridgehead_template_ids]
    if len(set(bridgehead_locants)) != 2:
        return None
    keyed = [(_locant_key(loc), loc) for loc in bridgehead_locants]
    if any(key is None for key, _ in keyed):
        return None
    bridgehead_locants = [loc for _, loc in sorted(keyed)]
    bridge_locant = _next_bridge_locant(locants)
    if bridge_locant is None:
        return None

    core_name = _core_name_on_original_graph(mol, template, tuple(match))
    if not core_name:
        return None
    indicated, unindicated_name = _without_indicated_hydrogen(core_name)
    _, unindicated_template = _without_indicated_hydrogen(name)
    if not unindicated_name.endswith(unindicated_template):
        return None
    pre_core = unindicated_name[:-len(unindicated_template)]
    # The hydro-state prefix is retained from the original mapped core and the
    # bridge term is inserted between that prefix and the source core hydride.
    bridge_name = ",".join(bridgehead_locants) + "-methano"
    rendered = indicated + "-" if indicated else ""
    rendered += pre_core + bridge_name + unindicated_template

    core_atom_locants = tuple((mol_id, str(locants[j])) for j, mol_id in enumerate(match))
    atom_locants = core_atom_locants + ((bridge_id, bridge_locant),)
    roles = (
        ("fused_core", core, core_atom_locants),
        ("methano_bridge", frozenset({bridge_id}), ((bridge_id, bridge_locant),)),
    )
    if (core & {bridge_id} or set().union(*(set(role[1]) for role in roles)) != set(heavy)
            or sum(len(role[1]) for role in roles) != len(heavy)):
        return None
    from .naming_methods import MethodDecision
    return MethodDecision(
        name=rendered, method="bridged-fused", principal_centers=frozenset(),
        atoms=heavy, rule=RULE, atom_locants=atom_locants,
        component_roles=roles,
    )


def select_methano_fused_decision(mol, Chem):
    """Discover a mapped one-methano retained fused parent on the input graph.

    P-25.4.3.4 first preserves an available complete fused construction. For
    the bounded bridge universe, criteria .4.2(a,b,d) compare core ring count,
    size and ring sizes before .4.2(i)/.3.3 compare bridge attachment locants.
    Distinct tied core vocabularies are not resolved by length or spelling.
    """
    from dataclasses import replace
    from .naming_context import MoleculeContext
    from .naming_result import SelectionLimitation
    from .parent_rules import _cached_iter_fused_template_matches
    from .ring_state_parent import _fused_templates

    if (not 7 <= mol.GetNumAtoms() <= 96 or mol.GetStereoGroups()
            or mol.GetNumBonds() - mol.GetNumAtoms() + 1 < 3
            or any(a.GetAtomicNum() != 6 or a.GetFormalCharge() or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or not a.IsInRing() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    bridges = [a.GetIdx() for a in mol.GetAtoms()
               if not a.GetIsAromatic() and a.GetDegree() == 2
               and a.GetTotalNumHs(includeNeighbors=True) == 2
               and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())]
    if not bridges:
        return None
    templates = _fused_templates(Chem)

    def matches(graph):
        ctx = MoleculeContext(graph, Chem)
        return _cached_iter_fused_template_matches(
            ctx, ctx.candidate_templates(templates), Chem, topology_query=True,
            require_one_ring_system=True, reject_extra_core_bonds=True)

    if any(len(match) == mol.GetNumAtoms() for _template, match in matches(mol)):
        return None  # P-25.4.3.4: ordinary fusion is available for the full graph.
    rows = []
    for bridge in bridges:
        original = tuple(i for i in range(mol.GetNumAtoms()) if i != bridge)
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(bridge)
        core_graph = rw.GetMol()
        try:
            Chem.SanitizeMol(core_graph)
        except (ValueError, RuntimeError):
            continue
        for template, core_match in matches(core_graph):
            if len(core_match) != len(original):
                continue
            query = template['mol']
            rings = tuple(tuple(ring) for ring in query.GetRingInfo().AtomRings())
            proof = FusedCoreProof(template['name'], rings)
            match = tuple(original[i] for i in core_match)
            decision = methano_fused_decision(mol, Chem, template, match, core_proof=proof)
            if decision is None:
                continue
            mapping = dict(decision.atom_locants)
            endpoints = tuple(sorted((_locant_key(mapping[n.GetIdx()])
                                      for n in mol.GetAtomWithIdx(bridge).GetNeighbors())))
            core_key = (-query.GetRingInfo().NumRings(), -query.GetNumAtoms(),
                        tuple(-size for size in sorted(map(len, rings), reverse=True)))
            rows.append((core_key, template['name'], endpoints, decision))
    if not rows:
        return None
    best_core = min(row[0] for row in rows)
    rows = [row for row in rows if row[0] == best_core]
    if len({row[1] for row in rows}) != 1:
        return None  # A missing P-44 ring-system criterion is not a tie-breaker.
    best_endpoints = min(row[2] for row in rows)
    choices = [row[3] for row in rows if row[2] == best_endpoints]
    if len({choice.name for choice in choices}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(choices, key=lambda row: tuple(sorted((locant, ranks[idx])
                                                   for idx, locant in row.atom_locants)))
    return replace(chosen, rule=chosen.rule + '/P-25.4.3.3/P-25.4.3.4.2',
                   preference_issues=(SelectionLimitation(
                       'ring-construction', 'P-25.4.3.4/P-25.4.3.4.2',
                       'Retained-core one-methano construction is mapped; comparison '
                       'with other fusion vocabularies and bridge decompositions remains unresolved.'),))
