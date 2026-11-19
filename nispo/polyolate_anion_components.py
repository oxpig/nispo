"""Mapped multi-site polyolate anions for a bounded alcohol-salt family.

This is an opt-in component producer. It does not alter public dispatch or
the existing ionic assembler. The first scope is an unbranched saturated
carbon parent bearing two or more terminal O− sites and optional residual OH
groups (P-72.2.2.2.2, P-77.2.1/.2).
"""

from dataclasses import dataclass


_PARENT_STEMS = {
    2: "ethane", 3: "propane", 4: "butane", 5: "pentane",
    6: "hexane", 7: "heptane", 8: "octane", 9: "nonane", 10: "decane",
}
_COMPOUND_MULTIPLIERS = {2: "bis", 3: "tris", 4: "tetrakis",
                         5: "pentakis", 6: "hexakis"}


@dataclass(frozen=True, slots=True)
class PolyolateAnionComponent:
    """A complete mapped polyolate component retaining its original charge."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    anionic_oxygens: tuple[int, ...]
    hydroxy_oxygens: tuple[int, ...]
    parent_atoms: frozenset[int]
    embedded_atoms: frozenset[int] = frozenset()


def polyolate_anion_component(mol, Chem):
    """Return a bounded multi-olate component, or ``None`` if unproved.

    The carbon skeleton must be a simple saturated acyclic chain of 2–10
    carbons. At least two terminal carbon-bound O atoms must each carry −1;
    any remaining O atoms must be neutral terminal OH groups. Every oxygen
    group must be attached to a parent carbon. Explicit hydrogen atoms,
    annotations, extra functions, and branching are outside this increment.
    """
    if (mol.GetNumHeavyAtoms() > 32 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()):
        return None
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() == 1 for a in atoms):
        return None
    if any(a.GetAtomicNum() not in {6, 8} or a.GetNumRadicalElectrons()
           or a.GetIsotope() or int(a.GetChiralTag()) for a in atoms):
        return None
    if any(b.GetStereo() or b.GetBondDir() != Chem.BondDir.NONE
           or b.GetBondType() != Chem.BondType.SINGLE
           for b in mol.GetBonds()):
        return None

    carbons = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() == 6)
    oxygens = tuple(a for a in atoms if a.GetAtomicNum() == 8)
    if not 2 <= len(carbons) <= 10 or not oxygens:
        return None
    if any(a.GetFormalCharge() != 0 for a in atoms if a.GetAtomicNum() == 6):
        return None

    adjacency = {idx: set() for idx in carbons}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomicNum() == right.GetAtomicNum() == 6:
            adjacency[left.GetIdx()].add(right.GetIdx())
            adjacency[right.GetIdx()].add(left.GetIdx())
        elif {left.GetAtomicNum(), right.GetAtomicNum()} != {6, 8}:
            return None
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return None
    ends = [idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1]
    if len(ends) != 2 or sum(map(len, adjacency.values())) != 2 * (len(carbons) - 1):
        return None
    path = []
    previous, current = None, ends[0]
    while True:
        path.append(current)
        following = [i for i in adjacency[current] if i != previous]
        if not following:
            break
        if len(following) != 1 or following[0] in path:
            return None
        previous, current = current, following[0]
    if frozenset(path) != carbons or any(mol.GetAtomWithIdx(i).IsInRing() for i in carbons):
        return None

    anionic, hydroxy = [], []
    for oxygen in oxygens:
        if oxygen.GetDegree() != 1:
            return None
        owner = oxygen.GetNeighbors()[0]
        if owner.GetAtomicNum() != 6 or mol.GetBondBetweenAtoms(
                oxygen.GetIdx(), owner.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
            return None
        if oxygen.GetFormalCharge() == -1 and oxygen.GetTotalNumHs(includeNeighbors=True) == 0:
            anionic.append(oxygen.GetIdx())
        elif oxygen.GetFormalCharge() == 0 and oxygen.GetTotalNumHs(includeNeighbors=True) == 1:
            hydroxy.append(oxygen.GetIdx())
        else:
            return None
    if len(anionic) < 2 or sum(a.GetFormalCharge() for a in atoms) != -len(anionic):
        return None
    if any(mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() not in carbons
           for i in (*anionic, *hydroxy)):
        return None

    from .constants import MULT
    from .formatting import join_prefix_parent
    from .parent_selection import ParentFacts, locant_key, locant_set
    from .anion_parent_selection import (
        AnionicCenterFact, AnionicParentCandidate, anion_parent_frontier,
    )

    count = len(anionic)
    multiplier = _COMPOUND_MULTIPLIERS.get(count)
    if not multiplier:
        return None
    candidates = []
    for oriented in (tuple(path), tuple(reversed(path))):
        locants = {idx: str(pos) for pos, idx in enumerate(oriented, 1)}
        suffix_locs = tuple(sorted(int(locants[mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()])
                                   for i in anionic))
        hydroxy_locs = tuple(sorted(int(locants[mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()])
                                    for i in hydroxy))
        prefix_text = ""
        if hydroxy_locs:
            hydroxy_multiplier = MULT.get(len(hydroxy), "")
            if len(hydroxy) > 1 and not hydroxy_multiplier:
                continue
            locant_text = ",".join(map(str, hydroxy_locs))
            prefix_text = f"{locant_text}-{hydroxy_multiplier}hydroxy"
        parent = _PARENT_STEMS[len(oriented)]
        suffix = f"{parent}-{','.join(map(str, suffix_locs))}-{multiplier}(olate)"
        name = join_prefix_parent(prefix_text, suffix) if prefix_text else suffix
        group_facts = tuple(
            AnionicCenterFact(
                # The typed anionic center owns the charged oxygen. Its carbon
                # host is separately proved by the matcher and belongs to the
                # parent, so two geminal O− groups do not overlap ownership.
                atom_ids=frozenset((i,)),
                charge_atom_ids=frozenset((i,)), class_symbol="O", operation="ate",
                evidence_kind="group", source_rule="P-72.2.2.2.2",
                proof="terminal O− center; matcher separately proves one single-bonded parent-carbon host",
            ) for i in anionic
        )
        parent_facts = ParentFacts(
            principal_class=(4,), suffix_count=len(anionic), kind="chain",
            elements=((6, len(oriented)),), ring_count=0, multiple_count=0,
            double_count=0, nonstandard_key=(), indicated_h_locants=(),
            replacement_locants=(), replacement_element_locants=(), fusion_locants=(),
            suffix_locants=locant_set(suffix_locs), attachment_locants=(),
            unsaturation_locants=(), double_locants=(), isotope_key=(), stereo_key=(),
            multiplied_parents=1, prefix_count=len(hydroxy),
            prefix_locants=locant_set(hydroxy_locs),
            prefix_citation_locants=tuple(locant_key(i) for i in hydroxy_locs),
            substituent_nonstandard_key=(), substituent_isotope_key=(),
            alphanumerical_key=(),
        )
        candidate = AnionicParentCandidate(
            identity="linear-carbon-polyolate-parent",
            parent_atom_ids=frozenset((*oriented, *anionic, *hydroxy)),
            considered_charge_atoms=frozenset(anionic),
            parent_centers=group_facts, suffix_key=(), parent_facts=parent_facts,
            proof="all carbon-bound anionic hydroxy centers are suffix centers on this parent",
        )
        candidates.append({"name": name, "path": oriented, "locants": locants,
                           "suffix_locs": suffix_locs, "hydroxy_locs": hydroxy_locs,
                           "candidate": candidate})
    if not candidates:
        return None
    frontier = anion_parent_frontier(mol, [row["candidate"] for row in candidates])
    frontier_ids = {id(row) for row in frontier}
    selected_rows = [row for row in candidates if id(row["candidate"]) in frontier_ids]
    if not selected_rows or len({row["name"] for row in selected_rows}) != 1:
        return None
    selected = min(selected_rows, key=lambda row: tuple(row["path"]))
    name, selected_path, mapping = selected["name"], selected["path"], selected["locants"]
    parent_atoms = frozenset(selected_path)
    oxygen_atoms = frozenset((*anionic, *hydroxy))
    component_atoms = parent_atoms | oxygen_atoms
    if component_atoms != frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() > 1):
        return None
    atom_locants = tuple((idx, mapping[idx]) for idx in selected_path)
    atom_locants += tuple((idx, mapping[mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetIdx()])
                          for idx in (*anionic, *hydroxy))
    if len({idx for idx, _ in atom_locants}) != len(atom_locants):
        return None
    return PolyolateAnionComponent(
        name=name,
        atoms=component_atoms,
        atom_locants=atom_locants,
        charge=-len(anionic),
        rule="P-72.2.2.2.2/P-77.2.1/P-77.2.2/P-44/P-45",
        proof=("linear saturated C2-C10 parent", "two or more terminal carbon-bound O− sites",
               "residual terminal OH groups rendered as hydroxy prefixes",
               "all original heavy atoms owned once", "original formal charge retained"),
        anionic_oxygens=tuple(sorted(anionic)),
        hydroxy_oxygens=tuple(sorted(hydroxy)),
        parent_atoms=parent_atoms,
    )
