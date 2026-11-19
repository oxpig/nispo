"""Mapped sulfonate and sulfinate organic-anion components.

This component producer supplies original graph ownership and charge to ionic
assembly. It does not construct multi-component salts.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SulfurAnionComponent:
    """A typed organic sulfur oxyanion, compatible with acid-salt components."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    identity: str


def _acyclic_carbon_parent(mol, carbon_atoms):
    """Prove that the component has one connected, acyclic C-only skeleton."""
    carbon_atoms = set(carbon_atoms)
    if not carbon_atoms:
        return False
    adjacency = {idx: set() for idx in carbon_atoms}
    edge_count = 0
    for idx in carbon_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing() or atom.GetIsAromatic():
            return False
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other not in carbon_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(idx, other)
            if bond.GetBondTypeAsDouble() not in {1.0, 2.0, 3.0}:
                return False
            adjacency[idx].add(other)
            if idx < other:
                edge_count += 1
    if edge_count != len(carbon_atoms) - 1:
        return False
    seen, stack = set(), [next(iter(carbon_atoms))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != carbon_atoms:
        return False
    # A carbon skeleton branch remains eligible when it is a substitutive
    # carbon parent; reject unusually high-degree carbon graphs here.
    return all(len(neighbors) <= 3 for neighbors in adjacency.values())


def _candidate_facts(mol, facts_mol, group, candidate):
    from .parent_selection import graph_parent_facts

    construction = candidate.construction
    if construction is None:
        return None
    parent_atoms = frozenset(candidate.parent_atoms)
    if not parent_atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                               for i in parent_atoms):
        return None
    owner_locants = dict(candidate.atom_locants)
    if group.owner not in parent_atoms or group.owner not in owner_locants:
        return None
    edge_count = sum(1 for bond in mol.GetBonds()
                     if bond.GetBeginAtomIdx() in parent_atoms
                     and bond.GetEndAtomIdx() in parent_atoms)
    if edge_count != len(parent_atoms) - 1:
        return None
    return graph_parent_facts(
        facts_mol, parent_atoms, dict(candidate.atom_locants),
        principal_class=group.seniority,
        suffix_centers=candidate.principal_centers,
        suffix_locants=candidate.numbering_key[2],
        prefixes=candidate.prefixes,
        kind="chain",
        name=candidate.name,
        allow_annotations=True,
    )


def sulfur_anion_component(mol, Chem):
    """Return one mapped P-72 sulfonate/sulfinate component, else ``None``.

    The bounded graph is one acyclic carbon parent directly bearing exactly
    one sulfonate or sulfinate group. Carbon-13 and carbon tetrahedral stereo
    annotations may be combined when the common mapped parent writer yields a
    complete, comparable original-graph construction. Other stereo forms are
    outside this component's supported grammar.
    """
    if (mol.GetNumHeavyAtoms() > 64 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()):
        return None
    if any(atom.GetFormalCharge() not in {0, -1} or atom.GetNumRadicalElectrons()
           for atom in mol.GetAtoms()):
        return None
    if any(atom.GetIsotope() and atom.GetAtomicNum() != 6
           for atom in mol.GetAtoms()):
        return None
    # P-82.4 describes the joint naming case and puts stereo descriptors
    # before isotope descriptors when both occupy the same name position.
    # The shared writer gets stereochemistry from the original, charged graph;
    # neutralization below remains solely a parent-fact comparison copy.
    if any(int(bond.GetStereo()) for bond in mol.GetBonds()):
        return None
    if any(int(atom.GetChiralTag()) and atom.GetAtomicNum() != 6
           for atom in mol.GetAtoms()):
        return None
    if any(atom.GetAtomicNum() not in {6, 8, 16}
           for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1):
        return None

    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    groups = tuple(inventory.groups)
    if (not inventory.supported or len(groups) != 1
            or len(inventory.principal) != 1
            or inventory.principal[0] != groups[0]
            or groups[0].kind not in {"sulfonate", "sulfinate"}):
        return None
    group = groups[0]
    if group.center < 0 or group.owner < 0 or group.ligand < 0:
        return None
    sulfur = mol.GetAtomWithIdx(group.center)
    carbon = mol.GetAtomWithIdx(group.owner)
    anionic_oxygen = mol.GetAtomWithIdx(group.ligand)
    if (sulfur.GetAtomicNum() != 16 or carbon.GetAtomicNum() != 6
            or anionic_oxygen.GetAtomicNum() != 8
            or anionic_oxygen.GetFormalCharge() != -1
            or sulfur.GetFormalCharge() != 0
            or sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != -1):
        return None
    sulfur_bonds = [(bond.GetOtherAtom(sulfur), bond.GetBondTypeAsDouble())
                    for bond in sulfur.GetBonds()]
    double_oxygens = [atom.GetIdx() for atom, order in sulfur_bonds
                      if atom.GetAtomicNum() == 8 and order == 2]
    single_oxygens = [atom.GetIdx() for atom, order in sulfur_bonds
                      if atom.GetAtomicNum() == 8 and order == 1]
    single_carbons = [atom.GetIdx() for atom, order in sulfur_bonds
                      if atom.GetAtomicNum() == 6 and order == 1]
    expected_double_count = 2 if group.kind == "sulfonate" else 1
    if (len(double_oxygens) != expected_double_count
            or len(single_oxygens) != 1 or single_oxygens[0] != group.ligand
            or len(single_carbons) != 1 or single_carbons[0] != group.owner
            or len(sulfur_bonds) != expected_double_count + 2):
        return None

    carbon_atoms = {atom.GetIdx() for atom in mol.GetAtoms()
                    if atom.GetAtomicNum() == 6}
    if not _acyclic_carbon_parent(mol, carbon_atoms):
        return None
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1)
    if set(carbon_atoms) | {group.center, *double_oxygens, group.ligand} != set(heavy):
        return None

    from .complete_functional_parent import functional_parent_candidates
    from .parent_selection import parent_frontier

    # Parent-fact comparison assumes a neutral naming graph. P-72 derives the
    # anion from its acid by removal of the terminal hydron; cap that same
    # mapped oxygen on a comparison copy only. Rendering and ownership still
    # use the charged input graph below.
    neutralized = Chem.RWMol(mol)
    cap = neutralized.GetAtomWithIdx(group.ligand)
    cap.SetFormalCharge(0)
    cap.SetNoImplicit(False)
    cap.SetNumExplicitHs(0)
    facts_mol = neutralized.GetMol()
    try:
        Chem.SanitizeMol(facts_mol)
    except (ValueError, RuntimeError):
        return None

    candidates = functional_parent_candidates(
        mol, Chem, include_single=True, principal=(group,)
    )
    eligible = []
    facts = {}
    for candidate in candidates:
        if candidate.principal_centers != frozenset({group.center}):
            continue
        construction = candidate.construction
        if construction is None:
            continue
        if frozenset(construction.atoms | construction.embedded_atoms) != heavy:
            continue
        if not set(candidate.parent_atoms) <= carbon_atoms:
            continue
        parent_facts = _candidate_facts(mol, facts_mol, group, candidate)
        if parent_facts is None:
            continue
        eligible.append(candidate)
        facts[id(candidate)] = parent_facts
    if not eligible:
        return None
    frontier = parent_frontier(eligible, lambda candidate: facts[id(candidate)])
    # Different surviving constructions are not interchangeable unless they
    # spell the same source-backed anion and map the same original parent.
    if len({candidate.name for candidate in frontier}) != 1:
        return None
    parent_scopes = {frozenset(candidate.parent_atoms) for candidate in frontier}
    if len(parent_scopes) != 1:
        return None
    chosen = min(frontier, key=lambda candidate: (
        tuple(candidate.atom_locants), tuple(candidate.prefixes), candidate.name
    ))
    if not chosen.name.endswith("sulfonate") and not chosen.name.endswith("sulfinate"):
        return None
    atom_locants = tuple((idx, str(locant)) for idx, locant in chosen.atom_locants)
    if not atom_locants or not set(idx for idx, _ in atom_locants) <= heavy:
        return None
    if not set(chosen.construction.atoms | chosen.construction.embedded_atoms) <= heavy:
        return None
    parent_role = frozenset(idx for idx in carbon_atoms)
    anion_role = frozenset({group.center, group.ligand, *double_oxygens})
    if parent_role & anion_role or parent_role | anion_role != heavy:
        return None
    identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    proof = (
        ("center", group.center), ("host", group.owner),
        ("anionic_oxygen", group.ligand), ("kind", group.kind),
        ("parent_atoms", tuple(sorted(chosen.parent_atoms))),
        ("component_roles", (("carbon-parent", parent_role),
                             ("sulfur-oxoanion", anion_role))),
        ("charge", -1),
        ("stereo", tuple(sorted((idx, int(atom.GetChiralTag()))
                                  for idx, atom in ((a.GetIdx(), a) for a in mol.GetAtoms())
                                  if atom.GetChiralTag()))),
        ("isotopes", tuple(sorted((atom.GetIdx(), atom.GetIsotope())
                                   for atom in mol.GetAtoms() if atom.GetIsotope()))),
    )
    return SulfurAnionComponent(
        chosen.name, heavy, atom_locants, -1,
        "P-65.3.1/P-72.2.2.2.1.1", proof, identity,
    )
