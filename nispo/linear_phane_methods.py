"""Bounded P-26 producer for linear phanes of identical aromatic amplificants.

The producer proves a path of isolated benzene/furan/thiophene/pyridine rings and ordinary-valence
singleton C/N/O/S linkers. It records the P-52.2.5.1 gate separately from its
explicit graph/frontier resource limits, so a proven mandatory phane family
stays pending when bounded rendering declines. It does not decide method
selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import prod

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT


_MAX_GRAPH_HEAVY_ATOMS = 96
_MAX_FULL_FRONTIER = 8192
_UNIT_TERMS = {1: "mono", 2: "di", 3: "tri", 4: "tetra", 5: "penta",
               6: "hexa", 7: "hepta", 8: "octa", 9: "nona"}
_TENS_TERMS = {2: "icosa", 3: "triaconta", 4: "tetraconta",
               5: "pentaconta", 6: "hexaconta", 7: "heptaconta",
               8: "octaconta", 9: "nonaconta"}
_HETERO_PREFIX = {8: "oxa", 16: "thia", 7: "aza"}
_HETERO_SENIORITY = (8, 16, 7)  # P-26.5.4.2 / P-15.4.1.2: O > S > N


@dataclass(frozen=True, slots=True)
class LinearPhaneDecision:
    """A complete mapped parent proposal, with the source gate retained."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    component_roles: tuple[tuple[str, tuple[int, ...], tuple[tuple[int, str], ...]], ...]
    rule: str
    simplified_node_count: int
    ring_count: int
    equivalent_atom_locant_maps: tuple[tuple[tuple[int, str], ...], ...]


@dataclass(frozen=True, slots=True)
class LinearPhaneFrontier:
    """Typed result separating P-52 eligibility from rendering coverage."""

    disposition: str  # complete, pending, unmatched
    mandatory_eligible: bool
    decision: LinearPhaneDecision | None
    source_rule: str
    reason: str
    simplified_node_count: int | None = None
    ring_count: int | None = None
    _proof_factory: object = field(default=None, repr=False, compare=False)

    @property
    def justification(self):
        """Compute scoped preference evidence only for structured delivery."""
        return self._proof_factory() if self._proof_factory is not None else None


@dataclass(frozen=True, slots=True)
class _PathProof:
    nodes: tuple[tuple[str, frozenset[int]], ...]
    adjacency: tuple[tuple[int, ...], ...]
    ring_nodes: frozenset[int]
    linker_nodes: frozenset[int]
    edge_atoms: tuple[tuple[int, int, int, int], ...]


def linear_phane_frontier(mol, Chem) -> LinearPhaneFrontier:
    """Assess the narrow path family without hiding mandatory pending cases.

    ``mandatory_eligible`` is true only when this producer proves the full
    P-52.2.5.1(2) structure gate. ``pending`` then means the source-preferred
    family is established but the bounded renderer did not complete it.
    ``unmatched`` means this limited producer did not prove that family; it
    makes no claim about other P-26 phane structures.
    """
    proof = _prove_path(mol, Chem)
    if proof is None:
        return LinearPhaneFrontier("unmatched", False, None,
                                   "P-52.2.5.1(2)",
                                   "the supported neutral identical-aromatic C/N/O/S path scope was not proved")
    path = _ordered_path(proof)
    if path is None:
        return LinearPhaneFrontier("unmatched", False, None,
                                   "P-52.2.5.1(2)", "a single linear simplified skeleton was not proved",
                                   len(proof.nodes), len(proof.ring_nodes))

    mandatory = (len(proof.ring_nodes) >= 4 and len(proof.nodes) >= 7
                 and path[0] in proof.ring_nodes and path[-1] in proof.ring_nodes)
    if not mandatory:
        return LinearPhaneFrontier("unmatched", False, None,
                                   "P-52.2.5.1(2)",
                                   "the published four-ring/two-terminal/seven-node gate is not met",
                                   len(proof.nodes), len(proof.ring_nodes))

    heavy_count = sum(1 for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1)
    if heavy_count > _MAX_GRAPH_HEAVY_ATOMS:
        return _pending(proof, "whole-graph rendering is bounded to 96 heavy atoms")

    # Source eligibility is established before bounded grammar checks. Any
    # later decline is an explicit pending obligation for the caller.
    if _phane_numerical_term(len(proof.nodes)) is None:
        return _pending(proof, "P-14.2 numerical term is not validated for this node count")
    if _basic_numerical_term(len(proof.ring_nodes)) is None:
        return _pending(proof, "P-14.2 multiplier is not validated for this amplificant count")
    if any(bond.GetBondDir() != Chem.BondDir.NONE or int(bond.GetStereo())
           for bond in iter_bonds(mol)):
        return _pending(proof, "directed-bond annotation is outside the bounded parent renderer")

    orientation_choices = []
    for order in (path, tuple(reversed(path))):
        ring_seqs = {}
        for node in proof.ring_nodes:
            attachments = _attachment_atoms(proof, node, order)
            seqs = _best_ring_numberings(mol, proof.nodes[node][1], attachments)
            if not seqs:
                return _pending(proof, "an intrinsic aromatic amplificant numbering could not be completed")
            ring_seqs[node] = seqs

        # Benzene's atom numbering is determined independently within each
        # amplificant. There are at most two tied orientations per unit after
        # minimizing its attachment locants; preserve those alternatives.
        choice_sets = [ring_seqs[node] for node in order if node in proof.ring_nodes]
        orientation_choices.append((order, choice_sets))

    full_frontier_size = sum(prod(len(options) for options in choices)
                             for _, choices in orientation_choices)
    if full_frontier_size > _MAX_FULL_FRONTIER:
        return _pending(proof, "complete P-26 numbering frontier exceeds the 8192-candidate resource cap")

    candidates = []
    for order, choice_sets in orientation_choices:
        for seq_choice in product(*choice_sets):
            local_by_node = dict(zip((node for node in order if node in proof.ring_nodes), seq_choice))
            candidate = _candidate_for_order(mol, proof, order, local_by_node)
            if candidate is None:
                return _pending(proof, "an eligible P-26 candidate could not be rendered")
            candidates.append(candidate)

    if not candidates:
        return _pending(proof, "P-26 numbering/name construction returned no complete candidate")
    best_key = min(row[0] for row in candidates)
    winners = [row for row in candidates if row[0] == best_key]
    # Remaining symmetry may map equivalent benzene atoms differently, but
    # it must not change the complete printed parent.
    if len({row[1] for row in winners}) != 1:
        return _pending(proof, "source-numbered frontier leaves different parent spellings")
    maps = tuple(sorted({row[2] for row in winners}))
    _, name, atom_locants, roles = min(winners, key=lambda row: row[2])
    atoms = frozenset(atom for _, group in proof.nodes for atom in group)
    if any({idx for idx, _ in mapping} != atoms for mapping in maps):
        return _pending(proof, "a source-equivalent candidate does not own every original heavy atom")
    decision = LinearPhaneDecision(
        name=name,
        atoms=atoms,
        atom_locants=atom_locants,
        component_roles=roles,
        rule="P-52.2.5.1(2)/P-52.2.5.3/P-26.1-P-26.5",
        simplified_node_count=len(proof.nodes),
        ring_count=len(proof.ring_nodes),
        equivalent_atom_locant_maps=maps,
    )
    from .phane_justification import lazy_ring_ether_justification
    proof_factory = lazy_ring_ether_justification(
        mol, Chem, proof, path, orientation_choices, candidates, winners, decision)
    return LinearPhaneFrontier("complete", True, decision,
                               "P-52.2.5.1(2)/P-52.2.5.3",
                               "mandatory eligibility and bounded P-26 rendering are complete",
                               len(proof.nodes), len(proof.ring_nodes), proof_factory)


def _pending(proof, reason):
    return LinearPhaneFrontier("pending", True, None,
                               "P-52.2.5.1(2)/P-52.2.5.3", reason,
                               len(proof.nodes), len(proof.ring_nodes))


def _prove_path(mol, Chem) -> _PathProof | None:
    if mol is None or mol.GetStereoGroups():
        return None
    # This necessary source gate precedes full atom inspection. It is already
    # required below; moving it here avoids scanning ordinary one-ring inputs.
    raw_rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(raw_rings)<4:return None
    for atom in iter_atoms(mol):
        if (atom.GetAtomicNum() not in {6, 7, 8, 16} or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()
                or int(atom.GetChiralTag())):
            return None
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope() for atom in iter_atoms(mol)):
        return None
    if any(bond.GetBondType() != Chem.BondType.SINGLE
           for bond in iter_bonds(mol) if not bond.IsInRing()):
        return None

    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    if any(left & right for i, left in enumerate(raw_rings) for right in raw_rings[i + 1:]):
        return None

    parents=tuple(_aromatic_parent(mol,ring) for ring in raw_rings)
    if None in parents or len(set(parents))!=1:return None
    for ring in raw_rings:
        internal = [bond for bond in iter_bonds(mol)
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring]
        if len(internal) != len(ring) or any(not bond.IsInRing()
                                     or not bond.GetIsAromatic() for bond in internal):
            return None
        if any(sum(neighbor.GetIdx() in ring
                   for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()) != 2 for i in ring):
            return None

    ring_union = set().union(*raw_rings)
    linker_atoms = heavy - ring_union
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8, 16}
           for i in linker_atoms):
        return None
    if any(sum(neighbor.GetIdx() in heavy for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
           for i in linker_atoms):
        return None
    for idx in linker_atoms:
        atom = mol.GetAtomWithIdx(idx)
        try:
            valence = atom.GetTotalValence()
            hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
        except Exception:
            return None
        element = atom.GetAtomicNum()
        if element == 6 and (valence != 4 or hydrogens != 2):
            return None
        if element == 7 and (valence != 3 or hydrogens != 1):
            return None
        if element in {8, 16} and (valence != 2 or hydrogens != 0):
            return None

    nodes = list(zip(parents,raw_rings))
    nodes.extend(("linker", frozenset((idx,))) for idx in sorted(linker_atoms))
    owner = {atom: node for node, (_, group) in enumerate(nodes) for atom in group}
    if set(owner) != heavy:
        return None
    adjacency = {node: [] for node in range(len(nodes))}
    edge_atoms = []
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left, right = owner.get(a), owner.get(b)
        if left == right:
            continue
        if left is None or right is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        adjacency[left].append(right)
        adjacency[right].append(left)
        edge_atoms.append((left, right, a, b))
    if len(edge_atoms) != len(nodes) - 1:
        return None
    if any(len(neighbors) not in {1, 2} for neighbors in adjacency.values()):
        return None
    terminals = {node for node, neighbors in adjacency.items() if len(neighbors) == 1}
    ring_nodes = frozenset(i for i, (kind, _) in enumerate(nodes) if kind != "linker")
    linker_nodes = frozenset(i for i, (kind, _) in enumerate(nodes) if kind == "linker")
    if len(terminals) != 2 or not terminals <= ring_nodes:
        return None
    # P-52.2.5.1 calls for two terminal rings; all singleton linker atoms are
    # internal nodes in this producer.
    return _PathProof(tuple(nodes), tuple(tuple(adjacency[i]) for i in range(len(nodes))),
                      ring_nodes, linker_nodes, tuple(edge_atoms))


def _ordered_path(proof: _PathProof) -> tuple[int, ...] | None:
    ends = [i for i, neighbors in enumerate(proof.adjacency) if len(neighbors) == 1]
    if len(ends) != 2:
        return None
    output = []
    previous, current = None, ends[0]
    while True:
        output.append(current)
        following = [n for n in proof.adjacency[current] if n != previous]
        if not following:
            break
        if len(following) != 1:
            return None
        previous, current = current, following[0]
        if current in output:
            return None
    if len(output) != len(proof.nodes) or output[-1] not in ends:
        return None
    return tuple(output)


def _edge_atom_for_node_pair(proof, left, right):
    for a, b, atom_a, atom_b in proof.edge_atoms:
        if (a, b) == (left, right):
            return atom_a, atom_b
        if (a, b) == (right, left):
            return atom_b, atom_a
    return None


def _attachment_atoms(proof, ring_node, order):
    position = {node: i for i, node in enumerate(order)}
    result = []
    for neighbor in proof.adjacency[ring_node]:
        pair = _edge_atom_for_node_pair(proof, ring_node, neighbor)
        if pair is None:
            return ()
        ring_atom, _ = pair
        result.append((position[neighbor], ring_atom))
    return tuple(atom for _, atom in sorted(result))


def _best_benzene_numberings(mol, ring, attachments):
    """Compatibility entry point for the original benzene numbering witness."""
    return _best_ring_numberings(mol,ring,attachments)


def _aromatic_parent(mol,ring):
    if any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):return None
    numbers=sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring)
    return {(6,6,6,6,6,6):'benzene',(6,6,6,6,8):'furan',
            (6,6,6,6,16):'thiophene',(6,6,6,6,6,7):'pyridine'}.get(tuple(numbers))


def _ring_sequences(mol,ring):
    parent=_aromatic_parent(mol,ring)
    if parent is None:return ()
    adjacency={i:tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring) for i in ring}
    if any(len(ns)!=2 for ns in adjacency.values()):return ()
    starts=ring if parent=='benzene' else tuple(i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum()!=6)
    sequences=set()
    for start in starts:
        for second in adjacency[start]:
            seq=[start,second]
            while len(seq)<len(ring):
                following=[i for i in adjacency[seq[-1]] if i!=seq[-2]]
                if len(following)!=1:break
                seq.append(following[0])
            if len(seq)==len(ring) and len(set(seq))==len(ring):sequences.add(tuple(seq))
    return tuple(sorted(sequences))


def _best_ring_numberings(mol, ring, attachments):
    if len(attachments) not in {1, 2} or len(set(attachments)) != len(attachments):
        return ()
    sequences = set()
    for sequence in _ring_sequences(mol,ring):
        locants=tuple(sequence.index(atom)+1 for atom in attachments)
        sequences.add(((tuple(sorted(locants)),locants),sequence))
    if not sequences:
        return ()
    minimum = min(key for key, _ in sequences)
    return tuple(sequence for key, sequence in sorted(sequences) if key == minimum)


def _candidate_for_order(mol, proof, order, local_by_node):
    skeleton_locant = {node: i + 1 for i, node in enumerate(order)}
    superatom_locants = tuple(sorted(skeleton_locant[node] for node in proof.ring_nodes))
    # P-26.4.2.2/.4: identical amplificants with differing attachment sets
    # are associated to increasing superatom locants, then compared in that
    # same order. P-26.3.2.2 fixes each pair's citation direction.
    attachment_groups = {}
    attachment_key = []
    directed_attachment_key = []
    for node in proof.ring_nodes:
        attachment_atoms = _attachment_atoms(proof, node, order)
        sequence = local_by_node[node]
        locants = tuple(sequence.index(atom) + 1 for atom in attachment_atoms)
        attachment_groups.setdefault(locants, []).append(skeleton_locant[node])
        attachment_key.append((skeleton_locant[node], tuple(sorted(locants))))
        directed_attachment_key.append((skeleton_locant[node],locants))
    attachment_key = tuple(value for _, value in sorted(attachment_key))
    directed_attachment_key = tuple(value for _,value in sorted(directed_attachment_key))

    # P-26.5.4.1/.2: compare all heteroatom primary locants first, then the
    # element-specific locant sets in seniority order (O > S > N).
    hetero_locants = {element: tuple(sorted(
        skeleton_locant[node] for node in proof.linker_nodes
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == element
               for i in proof.nodes[node][1])))
        for element in _HETERO_SENIORITY}
    all_hetero_locants = tuple(sorted(loc for values in hetero_locants.values()
                                      for loc in values))
    by_element_key = tuple(hetero_locants[element] for element in _HETERO_SENIORITY)
    # P-26.4.2.4 compares the directed attachment tuples in increasing
    # superatom order after their unordered sets tie. This closes e.g. the
    # remaining single unsymmetrical middle amplificant, before P-26.5.
    key = (superatom_locants, attachment_key,directed_attachment_key,all_hetero_locants, by_element_key)

    citation_groups = []
    for locants, superatoms in attachment_groups.items():
        citation_groups.append((min(superatoms), tuple(sorted(superatoms)), locants))
    citation_groups.sort()
    amp_locants = []
    for _, superatoms, locants in citation_groups:
        listed = ",".join(map(str, superatoms))
        if locants:
            listed += "(" + ",".join(map(str, locants)) + ")"
        amp_locants.append(listed)
    locant_prefix = ",".join(amp_locants)
    amplifier_term = _basic_numerical_term(len(proof.ring_nodes))
    suffix = _phane_numerical_term(len(order))
    if amplifier_term is None or suffix is None:
        return None
    parent=proof.nodes[next(iter(proof.ring_nodes))][0]
    amplificant=parent[:-1]+'a' if parent.endswith('e') else parent+'a'
    phane_parent = locant_prefix + "-" + amplifier_term + amplificant + suffix + "phane"

    hetero_prefix = _hetero_replacement_prefix(mol, proof, skeleton_locant)
    if hetero_prefix is None:
        return None
    name = hetero_prefix + phane_parent

    atom_locants = []
    roles = []
    for node in order:
        loc = skeleton_locant[node]
        kind, atoms = proof.nodes[node]
        if kind != "linker":
            sequence = local_by_node[node]
            mapped = tuple((atom, f"{loc}^{j + 1}") for j, atom in enumerate(sequence))
            roles.append((f"{kind}-amplificant:{loc}", tuple(sorted(atoms)), mapped))
        else:
            atom = next(iter(atoms))
            mapped = ((atom, str(loc)),)
            roles.append((f"simplified-node:{loc}", (atom,), mapped))
        atom_locants.extend(mapped)
    return key, name, tuple(atom_locants), tuple(roles)


def _basic_numerical_term(number: int) -> str | None:
    """Return a P-14.2.1 numerical term for 1..99, within local validation."""
    # Derive composite 20–99 terms from the source rule rather than relying
    # on the precomputed ROOTS/MULT vocabulary. This keeps the construction
    # correct at the icosa elision boundary and beyond the existing table.
    if 20 <= number <= 99:
        tens, units = divmod(number, 10)
        tens_term = _TENS_TERMS.get(tens)
        if tens_term is None:
            return None
        if units == 0:
            return tens_term
        unit_term = {1: "hen", 2: "do"}.get(units) or _UNIT_TERMS.get(units)
        if unit_term is None:
            return None
        if tens == 2 and units >= 2:
            # P-14.2.1.2: the i in icosa is elided after a vowel.
            tens_term = "cosa"
        return unit_term + tens_term
    if number in MULT:
        return MULT[number]
    if number in _UNIT_TERMS:
        return _UNIT_TERMS[number]
    if number == 10:
        return "deca"
    if number == 11:
        return "undeca"
    if 12 <= number <= 19:
        unit = number - 10
        prefix = {12: "do", 13: "tri", 14: "tetra", 15: "penta", 16: "hexa",
                  17: "hepta", 18: "octa", 19: "nona"}[number]
        return prefix + "deca"
    return None


def _phane_numerical_term(node_count: int) -> str | None:
    """Numerical term before `phane`; mono is never used in a formed name."""
    if node_count < 2:
        return None
    return _basic_numerical_term(node_count)


def _hetero_replacement_prefix(mol, proof, skeleton_locant) -> str | None:
    chunks = []
    for element in _HETERO_SENIORITY:
        prefix = _HETERO_PREFIX[element]
        locants = tuple(sorted(skeleton_locant[node] for node in proof.linker_nodes
                               if any(mol.GetAtomWithIdx(i).GetAtomicNum() == element
                                      for i in proof.nodes[node][1])))
        if not locants:
            continue
        multiplicative = "" if len(locants) == 1 else _basic_numerical_term(len(locants))
        if len(locants) > 1 and multiplicative is None:
            return None
        term = (multiplicative or "") + prefix
        chunks.append(",".join(map(str, locants)) + "-" + term)
    return "".join(chunk + "-" for chunk in chunks)
