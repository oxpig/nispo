"""Typed eligibility evidence for acyclic skeletal replacement (P-51.4.1).

This module proves only the acyclic replacement threshold and structural
conditions. It does not render a name. A caller must separately establish that
its parent spelling, suffixes, prefixes, annotations, and atom ownership are
supported by a producer.
"""

from dataclasses import dataclass


_TERMINAL_ELEMENTS = frozenset({6, 15, 33, 51, 83, 14, 32, 50, 82, 5, 13, 31, 49, 81})
_HETERO_ELEMENTS = frozenset({5, 7, 8, 13, 14, 15, 16, 31, 32, 33, 34,
                              49, 50, 51, 52, 81, 82, 83})


@dataclass(frozen=True, slots=True)
class Heterounit:
    """One source-defined replacement unit, preserving input atom indices."""

    atoms: tuple[int, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class ReplacementEligibility:
    """Eligibility proof or conservative decline for a proposed acyclic chain."""

    eligible: bool
    reason: str
    path: tuple[int, ...] = ()
    heterounits: tuple[Heterounit, ...] = ()
    principal_group_atoms: frozenset[int] = frozenset()

    @property
    def heterounit_count(self):
        return len(self.heterounits)


def acyclic_replacement_eligibility(mol, atoms, *, principal_group_atoms=(),
                                   has_principal_suffix=False):
    """Evaluate P-51.4.1.1–1.4 on an independently proposed parent chain.

    ``atoms`` is the candidate parent graph after class-specific functional
    groups and separately named organyl components have been assigned. Any
    principal-group heteroatom accidentally included in that candidate makes
    the proof fail. Unrecognized consecutive heteroatom groups return an
    explicit unknown result; they are never counted atom by atom by guess.
    """

    atoms = frozenset(atoms)
    principal_group_atoms = frozenset(principal_group_atoms)
    if not atoms:
        return ReplacementEligibility(False, "empty-parent-candidate",
                                      principal_group_atoms=principal_group_atoms)
    principal_heteroatoms = {i for i in principal_group_atoms
                             if mol.GetAtomWithIdx(i).GetAtomicNum() != 6}
    if atoms & principal_heteroatoms:
        return ReplacementEligibility(False, "principal-group-atoms-in-parent",
                                      principal_group_atoms=principal_group_atoms)
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return ReplacementEligibility(False, "cyclic-candidate",
                                      principal_group_atoms=principal_group_atoms)
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in _HETERO_ELEMENTS | {6}
           for i in atoms):
        return ReplacementEligibility(False, "unsupported-element-for-unit-proof",
                                      principal_group_atoms=principal_group_atoms)
    if any(mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms):
        return ReplacementEligibility(False, "charged-or-radical-parent-unsupported",
                                      principal_group_atoms=principal_group_atoms)

    degree = {i: sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())
              for i in atoms}
    if any(value > 2 for value in degree.values()):
        return ReplacementEligibility(False, "branched-candidate",
                                      principal_group_atoms=principal_group_atoms)
    ends = tuple(sorted(i for i, value in degree.items() if value <= 1))
    if len(ends) != 2:
        return ReplacementEligibility(False, "not-one-open-chain",
                                      principal_group_atoms=principal_group_atoms)

    path = []
    previous = None
    current = ends[0]
    while True:
        path.append(current)
        following = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                     if n.GetIdx() in atoms and n.GetIdx() != previous]
        if not following:
            break
        if len(following) != 1:
            return ReplacementEligibility(False, "not-one-open-chain",
                                          tuple(path), principal_group_atoms=principal_group_atoms)
        previous, current = current, following[0]
        if current in path:
            return ReplacementEligibility(False, "cyclic-candidate",
                                          tuple(path), principal_group_atoms=principal_group_atoms)
    if set(path) != atoms:
        return ReplacementEligibility(False, "disconnected-candidate",
                                      tuple(path), principal_group_atoms=principal_group_atoms)
    if not any(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path):
        return ReplacementEligibility(False, "chain-has-no-carbon",
                                      tuple(path), principal_group_atoms=principal_group_atoms)
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in _TERMINAL_ELEMENTS for i in ends):
        return ReplacementEligibility(False, "terminal-element-not-permitted",
                                      tuple(path), principal_group_atoms=principal_group_atoms)

    units, segment = [], []
    for idx in path:
        number = mol.GetAtomWithIdx(idx).GetAtomicNum()
        if number == 6:
            if segment:
                parsed = _heterounits(mol, segment, atoms,
                                      has_principal_suffix=has_principal_suffix)
                if parsed is None:
                    return ReplacementEligibility(False, "unknown-multivalent-heterounit",
                        tuple(path), tuple(units), principal_group_atoms)
                units.extend(parsed)
                segment = []
        else:
            segment.append(idx)
    if segment:
        parsed = _heterounits(mol, segment, atoms,
                              has_principal_suffix=has_principal_suffix)
        if parsed is None:
            return ReplacementEligibility(False, "unknown-multivalent-heterounit",
                tuple(path), tuple(units), principal_group_atoms)
        units.extend(parsed)

    if len(units) < 4:
        return ReplacementEligibility(False, "fewer-than-four-heterounits",
                                      tuple(path), tuple(units), principal_group_atoms)
    return ReplacementEligibility(True, "eligible-acyclic-replacement",
                                  tuple(path), tuple(units), principal_group_atoms)


def _heterounits(mol, segment, chain_atoms, *, has_principal_suffix=False):
    """Partition known named multivalent units, declining unknown clusters.

    P-51.4.1.1 names disulfanediyl, disiloxane-1,3-diyl, and
    dithioxanediyl as examples of one heterounit each. P-63.3.1 defines the
    O-O group as a peroxide bridge; this established multivalent group is
    retained as one unit. P-51.4.1.1 expressly treats oxysilanediyloxy and
    oxysulfanediyloxy as three consecutive units. Remaining single
    heteroatoms count as single-atom units.
    """

    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in segment)
    explicit_three = {
        (8, 14, 8),  # oxy-silanediyl-oxy: three units
        (8, 16, 8),  # oxy-sulfanediyl-oxy: three units
    }
    if numbers in explicit_three:
        return [Heterounit((i,), "heteroatom") for i in segment]

    grouped = {(8, 8): "peroxy",
               (16, 16): "disulfanediyl",
               (14, 8, 14): "disiloxane-1,3-diyl",
               (16, 8, 16): "dithioxanediyl"}
    if numbers in grouped:
        if (not _simple_group_context(mol, segment, chain_atoms)
                or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                       for a, b in zip(segment, segment[1:]))):
            return None
        return [Heterounit(tuple(segment), grouped[numbers])]

    # P-51.4.1.1 specifically states that, in the presence of a senior
    # suffix group, –O-P(O)(OCH3)-O– comprises three units. The candidate
    # chain contains O-P-O; the phosphoryl and methoxy oxygens are branches.
    if numbers == (8, 15, 8) and has_principal_suffix:
        phosphorus = mol.GetAtomWithIdx(segment[1])
        bridge_bonds_are_single = all(
            mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
            for a, b in zip(segment, segment[1:])
        )
        external_oxygens = [n for n in phosphorus.GetNeighbors()
            if n.GetIdx() not in {segment[0], segment[2]}
            and n.GetAtomicNum() == 8]
        phosphoryl = [n for n in external_oxygens
            if mol.GetBondBetweenAtoms(segment[1], n.GetIdx()).GetBondTypeAsDouble() == 2]
        alkoxy = [n for n in external_oxygens
            if mol.GetBondBetweenAtoms(segment[1], n.GetIdx()).GetBondTypeAsDouble() == 1
            and any(x.GetIdx() != segment[1] and x.GetAtomicNum() == 6
                    and x.GetDegree() == 1
                    and x.GetTotalNumHs(includeNeighbors=True) == 3
                    for x in n.GetNeighbors())]
        other = [n for n in phosphorus.GetNeighbors()
            if n.GetIdx() not in set(segment) | {x.GetIdx() for x in external_oxygens}
            and n.GetAtomicNum() > 1]
        if bridge_bonds_are_single and len(phosphoryl) == len(alkoxy) == 1 and not other:
            return [Heterounit((idx,), "heteroatom") for idx in segment]

    # P-21.2.2/.3 defines the simple parent-hydride vocabulary behind these
    # multivalent units. Unlike a homogeneous pair or a[ba] unit, a mixed
    # two-atom chain has no such parent name: e.g. Si-P and Si-O are two
    # consecutive units. P-51.4 prints both cases. Resolve only an unambiguous
    # partition; overlapping longer named parents remain outside this proof.
    if len(segment) > 1:
        return _ordinary_adjacent_units(mol, segment, chain_atoms)
    return [Heterounit((segment[0],), "heteroatom")]


def _ordinary_adjacent_units(mol, segment, chain_atoms):
    order = (8, 16, 34, 52, 7, 15, 33, 51, 83, 14, 32, 50, 82, 5, 13, 31, 49, 81)
    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in segment)
    # Nitrogen invokes amine/functional-parent rules in P-21.2.3.1. Boron
    # hydrides are expressly excluded from the homogeneous P-21.2.2 grammar.
    if 7 in numbers or any(a == b == 5 for a, b in zip(numbers, numbers[1:])):
        return None
    if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
           for a, b in zip(segment, segment[1:])):
        return None
    valences = {8: 2, 16: 2, 34: 2, 52: 2, 15: 3, 33: 3, 51: 3, 83: 3,
                14: 4, 32: 4, 50: 4, 82: 4, 5: 3, 13: 3, 31: 3, 49: 3, 81: 3}
    if any(mol.GetAtomWithIdx(i).GetTotalValence() != valences.get(number)
           for i, number in zip(segment, numbers)):
        return None
    windows = []
    for start in range(len(segment)):
        if start + 1 < len(segment) and numbers[start] == numbers[start + 1]:
            windows.append((start, start + 2, "homogeneous-divalent-unit"))
        if (start + 2 < len(segment) and numbers[start] == numbers[start + 2]
                and numbers[start] != numbers[start + 1]
                and order.index(numbers[start]) > order.index(numbers[start + 1])):
            windows.append((start, start + 3, "alternating-divalent-unit"))
    # Overlap includes a homogeneous chain of three or more atoms and longer
    # a[ba]n hydrides. Do not split those parents into short units to cross the
    # four-unit threshold. P-51.4 explicitly excludes trisulfane this way.
    if any(set(range(a, b)) & set(range(c, d))
           for pos, (a, b, _kind) in enumerate(windows)
           for c, d, _other in windows[pos + 1:]):
        return None
    starts = {start: (finish, kind) for start, finish, kind in windows}
    units, offset = [], 0
    while offset < len(segment):
        if offset in starts:
            finish, kind = starts[offset]
            atoms = segment[offset:finish]
            if not _simple_group_context(mol, atoms, chain_atoms):
                return None
            units.append(Heterounit(tuple(atoms), kind))
            offset = finish
        else:
            units.append(Heterounit((segment[offset],), "heteroatom"))
            offset += 1
    return units


def _simple_group_context(mol, segment, chain_atoms):
    """Reject named-group templates with unproved extra heavy-atom branches."""
    segment_set = set(segment)
    for offset, idx in enumerate(segment):
        expected = set()
        if offset:
            expected.add(segment[offset - 1])
        if offset + 1 < len(segment):
            expected.add(segment[offset + 1])
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1 or other in expected:
                continue
            if other not in chain_atoms or other in segment_set:
                return False
    return True
