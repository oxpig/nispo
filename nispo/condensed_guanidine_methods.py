"""Mapped PIN construction for unsubstituted condensed guanidines.

P-66.4.1.2.2 admits the diamides of imidodicarbonimidic through
triimidotetracarbonimidic acid for repeat counts two through four.  The
larger polyguanides use a-nomenclature instead.  This producer handles only
the explicitly depicted neutral, linear, unsubstituted C(=NH)-NH repeat
graphs; it does not infer tautomerism or substituent locants.
"""

from __future__ import annotations

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


_PARENT_NAMES = {
    2: "imidodicarbonimidic diamide",
    3: "diimidotricarbonimidic diamide",
    4: "triimidotetracarbonimidic diamide",
}
_RULE = "P-66.4.1.2.2"


def condensed_guanidine_decision(mol, Chem):
    """Return a complete decision for a source-defined condensed guanidine.

    All atom indices refer to ``mol``.  Locants are absent for the
    unsubstituted constructions; a single role records complete ownership.
    Unsupported annotations and any atom/bond outside the displayed linear
    repeat formula cause a conservative decline.
    """
    if mol is None or mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(atom.GetAtomicNum() == 1
           or atom.GetAtomicNum() not in {6, 7}
           or atom.GetFormalCharge() != 0
           or atom.GetNumRadicalElectrons()
           or atom.GetIsotope()
           or int(atom.GetChiralTag())
           for atom in iter_atoms(mol)):
        return None
    if any(int(bond.GetStereo()) or int(bond.GetBondDir())
           for bond in iter_bonds(mol)):
        return None

    carbons = [a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6]
    if len(carbons) not in {2, 3, 4}:
        return None
    n = len(carbons)
    imino_n = set()
    single_n = set()
    carbon_to_bridge = {}

    # Each repeat carbon has exactly one imino N and two single-bond N
    # neighbors. There are no carbon atoms outside these repeat centers.
    for cidx in carbons:
        carbon = mol.GetAtomWithIdx(cidx)
        doubles, singles = [], []
        for neighbor in carbon.GetNeighbors():
            other = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, other)
            if neighbor.GetAtomicNum() != 7:
                return None
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                doubles.append(other)
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                singles.append(other)
            else:
                return None
        if len(doubles) != 1 or len(singles) != 2:
            return None
        d = doubles[0]
        if d in imino_n:
            return None
        imino_n.add(d)
        single_n.update(singles)
        carbon_to_bridge[cidx] = tuple(singles)

    if len(imino_n) != n or len(single_n) != n + 1:
        return None
    if len(imino_n | single_n) + n != mol.GetNumHeavyAtoms():
        return None

    # Imino nitrogens are terminal =NH. Single-bonded nitrogens are either
    # terminal NH2 or an NH bridge shared by exactly two repeat carbons.
    for nidx in imino_n:
        atom = mol.GetAtomWithIdx(nidx)
        if atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
            return None
        neighbor = atom.GetNeighbors()[0]
        if neighbor.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(
                nidx, neighbor.GetIdx()).GetBondType() != Chem.BondType.DOUBLE:
            return None
    n_to_c = {idx: [] for idx in single_n}
    for cidx, neighbors in carbon_to_bridge.items():
        for nidx in neighbors:
            if nidx not in n_to_c:
                return None
            n_to_c[nidx].append(cidx)
    terminal_n, bridge_edges = [], []
    for nidx, attached in n_to_c.items():
        atom = mol.GetAtomWithIdx(nidx)
        if len(attached) == 1:
            if atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 2:
                return None
            terminal_n.append(nidx)
        elif len(attached) == 2:
            if atom.GetDegree() != 2 or atom.GetTotalNumHs(includeNeighbors=True) != 1:
                return None
            bridge_edges.append(tuple(attached))
        else:
            return None
    if len(terminal_n) != 2 or len(bridge_edges) != n - 1:
        return None

    # The carbon/bridge quotient must be one path, not a branched or cyclic
    # polyguanide. This graph test is independent of atom ordering.
    adjacency = {cidx: set() for cidx in carbons}
    for left, right in bridge_edges:
        if right in adjacency[left]:
            return None
        adjacency[left].add(right)
        adjacency[right].add(left)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    if sum(len(neighbors) == 1 for neighbors in adjacency.values()) != 2:
        return None
    seen, stack = set(), [carbons[0]]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    if seen != set(carbons):
        return None

    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if len(heavy) != 3 * n + 1:
        return None
    role = ("condensed-guanidine-parent", heavy, ())
    return MethodDecision(
        _PARENT_NAMES[n], "substitutive", frozenset(carbons), heavy,
        _RULE, atom_locants=(), component_roles=(role,))
