"""Scoped evidence for unannotated furan/pyridine ether-path phanes.

This does not infer a preferred name from a rendered string. The witness
checks the original graph, the two actual simplified-skeleton orientations,
their intrinsic ring numberings and source-ordered comparison keys. It is
computed lazily only if a structured result asks for preference evidence.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import prod

from .selection_evidence import PreferredNameJustification, SelectionStep


def _uniform_ring_ether_path(mol, proof, path):
    families = {proof.nodes[node][0] for node in proof.ring_nodes}
    if len(families) != 1 or not families <= {"furan", "pyridine"}:
        return False
    ring_positions = [i for i, node in enumerate(path) if node in proof.ring_nodes]
    if len(ring_positions) < 4 or ring_positions[0] != 0 or ring_positions[-1] != len(path) - 1:
        return False
    linkers = []
    for start, end in zip(ring_positions, ring_positions[1:]):
        segment = path[start + 1:end]
        if any(node not in proof.linker_nodes for node in segment):
            return False
        numbers = tuple(mol.GetAtomWithIdx(next(iter(proof.nodes[node][1]))).GetAtomicNum()
                        for node in segment)
        if len(numbers) < 3 or numbers[0] != 8 or numbers[-1] != 8 or any(
                number != 6 for number in numbers[1:-1]):
            return False
        linkers.append(numbers)
    return bool(linkers) and len(set(linkers)) == 1


def lazy_ring_ether_justification(mol, Chem, proof, path, orientations, candidates, winners, decision):
    """Return a memoized proof factory only for the admitted path topology."""
    if not _uniform_ring_ether_path(mol, proof, path):
        return None
    orientations = tuple((tuple(order), tuple(tuple(group) for group in choices))
                         for order, choices in orientations)
    candidates, winners = tuple(candidates), tuple(winners)

    @lru_cache(maxsize=1)
    def build():
        return _justify(mol, Chem, proof, tuple(path), orientations, candidates, winners, decision)

    return build


def _justify(mol, Chem, proof, path, orientations, candidates, winners, decision):
    from .linear_phane_methods import _attachment_atoms, _ring_sequences

    # Assigned, enhanced, isotopic and potentially unassigned stereo must not
    # be silently certified as absent. Use a copy for the RDKit perception.
    if mol.GetStereoGroups() or any(a.GetIsotope() or int(a.GetChiralTag())
                                    for a in mol.GetAtoms()):
        return None
    if any(b.GetBondDir() != Chem.BondDir.NONE or int(b.GetStereo()) for b in mol.GetBonds()):
        return None
    try:
        if Chem.FindPotentialStereo(Chem.Mol(mol)):
            return None
    except (ValueError, RuntimeError):
        return None

    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    expected_total = sum(prod(len(choices) for choices in choice_sets)
                         for _order, choice_sets in orientations)
    if (decision.atoms != heavy or len(orientations) != 2
            or not expected_total or len(candidates) != expected_total):
        return None
    if tuple(order for order, _ in orientations) != (path, tuple(reversed(path))):
        return None

    actual_comparisons = []
    local_census = []
    position = 0
    for order, choice_sets in orientations:
        ring_nodes = tuple(node for node in order if node in proof.ring_nodes)
        if len(ring_nodes) != len(choice_sets):
            return None
        attachment_rows, census = [], []
        for node, choices in zip(ring_nodes, choice_sets, strict=True):
            ring = proof.nodes[node][1]
            attachments = _attachment_atoms(proof, node, order)
            sequences = _ring_sequences(mol, ring)
            if len(sequences) != 2 or not choices or not attachments:
                return None
            keyed = [(tuple(sorted(sequence.index(atom) + 1 for atom in attachments)),
                      tuple(sequence.index(atom) + 1 for atom in attachments), sequence)
                     for sequence in sequences]
            best = min((unordered, directed) for unordered, directed, _ in keyed)
            expected = {sequence for unordered, directed, sequence in keyed
                        if (unordered, directed) == best}
            if set(choices) != expected:
                return None
            attachment_rows.append(best)
            census.append((len(sequences), len(choices)))

        superatom_locs = tuple(i + 1 for i, node in enumerate(order) if node in proof.ring_nodes)
        oxygen_locs = tuple(i + 1 for i, node in enumerate(order) if node in proof.linker_nodes
                            and mol.GetAtomWithIdx(next(iter(proof.nodes[node][1]))).GetAtomicNum() == 8)
        expected_key = (superatom_locs, tuple(row[0] for row in attachment_rows),
                        tuple(row[1] for row in attachment_rows), oxygen_locs,
                        (oxygen_locs, (), ()))
        for sequence_choices in product(*choice_sets):
            candidate = candidates[position]
            position += 1
            by_node = dict(zip(ring_nodes, sequence_choices, strict=True))
            key, _name, atom_locants, roles = candidate
            if key != expected_key:
                return None
            actual_map = dict(atom_locants)
            if (len(actual_map) != len(atom_locants) or set(actual_map) != heavy
                    or len(set(actual_map.values())) != len(heavy)):
                return None
            expected_map = {}
            for pos, node in enumerate(order, 1):
                if node in proof.ring_nodes:
                    expected_map.update((atom, f"{pos}^{local}")
                                        for local, atom in enumerate(by_node[node], 1))
                else:
                    expected_map[next(iter(proof.nodes[node][1]))] = str(pos)
            if actual_map != expected_map:
                return None
            role_atoms = [frozenset(atoms) for _role, atoms, _mapping in roles]
            if (not role_atoms or frozenset().union(*role_atoms) != heavy
                    or sum(map(len, role_atoms)) != len(heavy)
                    or any(dict(mapping) != {idx: actual_map[idx] for idx in atoms}
                           for _role, atoms, mapping in roles)):
                return None
            actual_comparisons.append(key)
        local_census.append(tuple(census))

    best_key = min(actual_comparisons)
    expected_winners = tuple(candidate for candidate in candidates if candidate[0] == best_key)
    if (winners != expected_winners or not winners
            or {row[1] for row in winners} != {decision.name}
            or set(decision.equivalent_atom_locant_maps) != {row[2] for row in winners}):
        return None
    comparisons = tuple(
        ("P-26.4.1/P-26.4.2/P-26.5.4", i + 1, key,
         "selected-or-tied" if key == best_key else "later")
        for i, key in enumerate(actual_comparisons)
    ) + (("intrinsic-ring-numbering-census", tuple(local_census)),)
    steps = (
        SelectionStep("principal-group", "P-41/P-52.2.5.3", "not-applicable",
                      "The complete furan or pyridine ether path has no suffix-bearing characteristic group."),
        SelectionStep("method", "P-52.2.5.1(2)/P-52.2.5.3", "proved",
                      "The complete linear skeleton has at least four rings, terminal rings and seven nodes."),
        SelectionStep("candidate-completeness", "P-26.4.1.2/P-26.4.2",
                      "proved", "Both skeleton orientations and every intrinsic ring numbering were examined."),
        SelectionStep("parent-and-numbering", "P-26.3.2/P-26.4/P-26.5.4", "proved",
                      "The actual ordered source keys select the delivered parent and its tied original-atom maps."),
        SelectionStep("isotope", "P-82", "not-applicable", "The original graph has no isotope labels."),
        SelectionStep("stereochemistry", "P-9", "not-applicable",
                      "The original graph has no assigned or RDKit-potential stereogenic feature."),
        SelectionStep("atom-ownership", "P-26.4.3.1", "proved",
                      "Every original heavy atom has one checked component and one candidate locant."),
    )
    return PreferredNameJustification(
        "unannotated-uniform-furan-or-pyridine-ether-linear-phane", len(candidates),
        expected_total, len(candidates), steps, comparisons, decision.name,
    )
