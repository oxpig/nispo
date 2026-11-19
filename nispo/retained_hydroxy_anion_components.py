"""Exact retained vocabulary for a source-listed alcohol-derived anion set.

This producer covers only the unmodified, unsubstituted anion structures
explicitly listed as preferred names in P-72.2.2.2.2/P-63.8.1. It is a
component vocabulary operation; salt assembly remains a separate concern.
"""

from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds


RULE = "P-72.2.2.2.2/P-77.2.1/P-63.8.1"
_RETAINED_NAMES = {
    "methoxide": "methoxide",
    "ethoxide": "ethoxide",
    "n-propoxide": "propoxide",
    "n-butoxide": "butoxide",
    "tert-butoxide": "tert-butoxide",
    "phenoxide": "phenoxide",
}


@dataclass(frozen=True, slots=True)
class RetainedHydroxyAnionComponent:
    """A mapped, charge-preserving retained alcohol/phenol anion component."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str | None], ...]
    atom_roles: tuple[tuple[str, tuple[int, ...]], ...]
    charge: int
    full_coverage: bool
    status: str
    origin: str
    rule: str
    proof: tuple
    anionic_oxygen: int
    oxygen_host: int
    parent_atoms: frozenset[int]
    substituted: bool = False


@dataclass(frozen=True, slots=True)
class RetainedHydroxyAnionOutcome:
    """Vocabulary-specific outcome; pending does not block another writer."""

    status: str
    component: RetainedHydroxyAnionComponent | None = None
    reason: str = ""


def _carbon_parent(mol, Chem):
    """Return a simple all-single acyclic carbon path, or None."""
    carbons = frozenset(a.GetIdx() for a in iter_atoms(mol)
                        if a.GetAtomicNum() == 6)
    if not carbons:
        return None
    adjacency = {idx: set() for idx in carbons}
    edges = 0
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomicNum() == right.GetAtomicNum() == 6:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            adjacency[left.GetIdx()].add(right.GetIdx())
            adjacency[right.GetIdx()].add(left.GetIdx())
            edges += 1
        elif left.GetAtomicNum() == right.GetAtomicNum() == 8:
            return None
        elif {left.GetAtomicNum(), right.GetAtomicNum()} == {6, 8}:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
        else:
            return None
    if edges != len(carbons) - 1 or any(len(row) > 3 for row in adjacency.values()):
        return None
    seen, stack = set(), [next(iter(carbons))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != carbons or any(mol.GetAtomWithIdx(i).IsInRing() for i in carbons):
        return None
    if len(carbons) == 1:
        return (next(iter(carbons)),)
    ends = [idx for idx, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) == 2 and all(len(row) <= 2 for row in adjacency.values()):
        return _walk_path(adjacency, ends[0])
    return None


def _walk_path(adjacency, start):
    path, previous, current = [], None, start
    while True:
        if current in path:
            return None
        path.append(current)
        following = [idx for idx in adjacency[current] if idx != previous]
        if not following:
            return tuple(path)
        if len(following) != 1:
            return None
        previous, current = current, following[0]


def _unsubstituted_phenoxide_parent(mol, Chem, oxygen_idx, carbons):
    """Recognize the exact six-carbon phenoxide ring and choose a map."""
    if len(carbons) != 6:
        return None
    if any(not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in carbons):
        return None
    adjacency = {idx: set() for idx in carbons}
    edges = 0
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in carbons and right in carbons:
            if bond.GetBondType() != Chem.BondType.AROMATIC:
                return None
            adjacency[left].add(right)
            adjacency[right].add(left)
            edges += 1
        elif oxygen_idx in (left, right) and (left if right == oxygen_idx else right) in carbons:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
    host = next((n.GetIdx() for n in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()
                 if n.GetAtomicNum() == 6), None)
    if host not in carbons or len(adjacency[host]) != 2 or edges != 6:
        return None
    if any(len(row) != 2 for row in adjacency.values()):
        return None
    # The ring is symmetric, and the retained term prints no ring locants.
    # Either neighbor gives a complete map without affecting the source name.
    first = min(adjacency[host])
    path = [host, first]
    previous, current = host, first
    while len(path) < 6:
        following = [i for i in adjacency[current] if i != previous]
        if len(following) != 1 or following[0] in path:
            return None
        previous, current = current, following[0]
        path.append(current)
    if host not in adjacency[path[-1]]:
        return None
    return tuple(path)


def retained_hydroxy_anion_component(mol, Chem) -> RetainedHydroxyAnionOutcome:
    """Recognize exact retained anions, returning a mapped component.

    A vocabulary-specific pending result means the graph otherwise has a
    listed retained skeleton but carries isotope labels. Callers must continue
    to any independently proved systematic writer; this result is not a
    whole-name empty-result instruction.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 32
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()):
        return RetainedHydroxyAnionOutcome("outside-scope", reason="not-one-bounded-component")
    if any(a.GetAtomicNum() == 7 for a in iter_atoms(mol)):
        component = _aminoxide_component(mol, Chem)
        if component is not None:
            return RetainedHydroxyAnionOutcome("complete", component)
    atoms = tuple(iter_atoms(mol))
    if (any(a.GetAtomicNum() == 1 for a in atoms)
            or any(a.GetAtomicNum() not in {6, 8} for a in atoms)
            or any(a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol))):
        return RetainedHydroxyAnionOutcome("outside-scope", reason="unsupported-atom-or-stereo")
    if sum(a.GetFormalCharge() for a in atoms) != -1:
        return RetainedHydroxyAnionOutcome("outside-scope", reason="not-monoanionic")
    oxygens = tuple(a for a in atoms if a.GetAtomicNum() == 8)
    if len(oxygens) != 1:
        return RetainedHydroxyAnionOutcome("outside-scope", reason="not-one-oxygen")
    oxygen = oxygens[0]
    if (oxygen.GetFormalCharge() != -1 or oxygen.GetDegree() != 1
            or oxygen.GetTotalNumHs(includeNeighbors=True) != 0):
        return RetainedHydroxyAnionOutcome("outside-scope", reason="not-terminal-O-anion")
    host = oxygen.GetNeighbors()[0]
    if host.GetAtomicNum() != 6 or host.GetFormalCharge() != 0:
        return RetainedHydroxyAnionOutcome("outside-scope", reason="unsupported-oxygen-host")
    if any(a.GetFormalCharge() for a in atoms if a.GetAtomicNum() != 8):
        return RetainedHydroxyAnionOutcome("outside-scope", reason="charge-not-localized-on-O")

    carbon_ids = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() == 6)
    parent_path = _carbon_parent(mol, Chem)
    isotope_present = any(a.GetIsotope() for a in atoms)
    if not isotope_present:
        name = None
        parent_order = None
        atom_roles = ()
        unlocanted_tert_butoxide = False
        if parent_path is not None and len(parent_path) in {1, 2, 3, 4}:
            if len(parent_path) == 1:
                name, parent_order = _RETAINED_NAMES["methoxide"], parent_path
            else:
                adjacency = {
                    idx: {n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                          if n.GetAtomicNum() == 6}
                    for idx in carbon_ids
                }
                # The path origin must be the oxygen-bearing terminal carbon.
                if host.GetIdx() in parent_path and len(adjacency[host.GetIdx()]) <= 1:
                    parent_order = _walk_path(adjacency, host.GetIdx())
                    name_by_length = {2: "ethoxide", 3: "propoxide", 4: "butoxide"}
                    if parent_order and len(parent_order) == len(carbon_ids):
                        name = name_by_length[len(parent_order)]
        if parent_path is None and len(carbon_ids) == 4:
            # Exact tert-butoxide: one central carbon with three methyl arms.
            neighbors = [n for n in host.GetNeighbors() if n.GetAtomicNum() == 6]
            if (host.GetIdx() in carbon_ids and len(neighbors) == 3
                    and all(len([x for x in n.GetNeighbors() if x.GetAtomicNum() == 6]) == 1
                            for n in neighbors)
                    and all(mol.GetBondBetweenAtoms(host.GetIdx(), n.GetIdx()).GetBondType()
                            == Chem.BondType.SINGLE for n in neighbors)):
                methyls = tuple(sorted(n.GetIdx() for n in neighbors))
                parent_order = None
                atom_roles = (
                    ("tert-butyl-central-carbon", (host.GetIdx(),)),
                    ("equivalent-methyl-carbons", methyls),
                    ("anionic-oxygen", (oxygen.GetIdx(),)),
                )
                unlocanted_tert_butoxide = True
                name = _RETAINED_NAMES["tert-butoxide"]
        if name is None and len(carbon_ids) == 6:
            parent_order = _unsubstituted_phenoxide_parent(
                mol, Chem, oxygen.GetIdx(), carbon_ids,
            )
            if parent_order is not None:
                name = _RETAINED_NAMES["phenoxide"]
        if name is not None and (parent_order is not None or unlocanted_tert_butoxide):
            if unlocanted_tert_butoxide:
                atom_locants = tuple((idx, None) for idx in (*carbon_ids, oxygen.GetIdx()))
            else:
                locants = {idx: str(position)
                           for position, idx in enumerate(parent_order, 1)}
                atom_locants = tuple((idx, locants[idx]) for idx in parent_order)
                atom_locants += ((oxygen.GetIdx(), locants[host.GetIdx()]),)
                if name == "phenoxide":
                    atom_roles = (
                        ("phenyl-ring", tuple(parent_order)),
                        ("anionic-oxygen", (oxygen.GetIdx(),)),
                    )
                else:
                    atom_roles = (
                        ("linear-carbon-parent", tuple(parent_order)),
                        ("anionic-oxygen", (oxygen.GetIdx(),)),
                    )
            heavy = frozenset((*carbon_ids, oxygen.GetIdx()))
            if (frozenset(idx for idx, _ in atom_locants) != heavy
                    or sum(a.GetFormalCharge() for a in atoms) != -1):
                return RetainedHydroxyAnionOutcome(
                    "outside-scope", reason="incomplete-retained-component-map",
                )
            component = RetainedHydroxyAnionComponent(
                name=name, atoms=heavy, atom_locants=atom_locants, charge=-1,
                atom_roles=atom_roles,
                full_coverage=True,
                status="complete", origin="source-listed-retained-name",
                rule=RULE,
                proof=("exact listed unsubstituted O-anion structure",
                       "single terminal O− attached to the listed carbon framework",
                       "all original heavy atoms mapped once",
                       "input formal charge retained"),
                anionic_oxygen=oxygen.GetIdx(), oxygen_host=host.GetIdx(),
                parent_atoms=carbon_ids,
            )
            return RetainedHydroxyAnionOutcome("complete", component)

    if isotope_present and _retained_skeleton_without_labels(mol, Chem):
        return RetainedHydroxyAnionOutcome(
            "pending", reason="isotope-bearing-retained-skeleton-needs-systematic-route",
        )
    return RetainedHydroxyAnionOutcome(
        "outside-scope", reason="not-a-listed-retained-anion-structure",
    )


def _aminoxide_component(mol, Chem):
    """P-72.2.2.2.2 retained H2N-O− and carbon-only N substitutions."""
    atoms = tuple(iter_atoms(mol))
    if (any(a.GetAtomicNum() not in {6, 7, 8} or a.GetIsotope()
            or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery() for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in iter_bonds(mol))):
        return None
    oxygens = [a for a in atoms if a.GetAtomicNum() == 8]
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    if len(oxygens) != 1 or len(nitrogens) != 1:
        return None
    oxygen, nitrogen = oxygens[0], nitrogens[0]
    if (oxygen.GetFormalCharge() != -1 or oxygen.GetDegree() != 1
            or oxygen.GetNeighbors()[0].GetIdx() != nitrogen.GetIdx()
            or nitrogen.GetFormalCharge() or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3
            or mol.GetBondBetweenAtoms(oxygen.GetIdx(), nitrogen.GetIdx()).GetBondTypeAsDouble() != 1
            or any(a.GetFormalCharge() for a in atoms if a.GetIdx() != oxygen.GetIdx())):
        return None
    roots = [a.GetIdx() for a in nitrogen.GetNeighbors() if a.GetIdx() != oxygen.GetIdx()]
    if len(roots) > 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
        return None
    from .functional_replacement_acids import _organyl_units

    seen = {oxygen.GetIdx(), nitrogen.GetIdx()}
    prefix = _organyl_units(mol, Chem, nitrogen.GetIdx(), roots, seen)
    heavy = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() > 1)
    if prefix is None or frozenset(seen) != heavy:
        return None
    return RetainedHydroxyAnionComponent(
        name=prefix + 'aminoxide', atoms=heavy,
        atom_locants=tuple((i, None) for i in sorted(heavy)),
        atom_roles=(('aminoxide-parent', (nitrogen.GetIdx(), oxygen.GetIdx())),
                    ('nitrogen-substituents', tuple(sorted(heavy - {nitrogen.GetIdx(), oxygen.GetIdx()})))),
        charge=-1, full_coverage=True, status='complete', origin='source-listed-retained-name',
        rule='P-72.2.2.2.2/P-63.8.1',
        proof=('terminal O− attached to neutral N; all N substituents are owned carbon groups',),
        anionic_oxygen=oxygen.GetIdx(), oxygen_host=nitrogen.GetIdx(),
        parent_atoms=heavy - {oxygen.GetIdx()}, substituted=bool(roots),
    )


def _retained_skeleton_without_labels(mol, Chem):
    """Check a listed constitution after clearing isotope metadata only."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
    outcome = retained_hydroxy_anion_component(copy, Chem)
    return outcome.status == "complete"
