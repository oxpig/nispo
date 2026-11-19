"""Bounded P-65.6.3.3.4.1 naming for a simple alternating polyester path.

This producer recognizes the source's two-identical-dicarboxylate / one
divalent-diol-linker / two-monovalent-alcohol pattern. Other assemblies remain
with their established routes or decline until their source construction is
implemented explicitly.
"""

import re

from .molecule_iteration import iter_atoms
from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision


def _fragments(mol, Chem):
    mappings = []
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True,
                                  fragsMolAtomMapping=mappings)
    return tuple((fragment, tuple(mapping)) for fragment, mapping in zip(fragments, mappings))


def _carbon_path(mol, atoms):
    """Return a saturated acyclic carbon path, or None."""
    atoms = set(atoms)
    if not atoms or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
        return None
    adjacency = {i: set() for i in atoms}
    for i in atoms:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j in atoms:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond.GetBondTypeAsDouble() != 1:
                    return None
                adjacency[i].add(j)
    seen, stack = set(), [next(iter(atoms))]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(adjacency[i] - seen)
    if seen != atoms or sum(map(len, adjacency.values())) // 2 != len(atoms) - 1:
        return None
    if len(atoms) == 1:
        return (next(iter(atoms)),)
    ends = [i for i in atoms if len(adjacency[i]) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in adjacency.values()):
        return None
    path = [ends[0]]
    while len(path) < len(atoms):
        next_atoms = adjacency[path[-1]] - set(path)
        if len(next_atoms) != 1:
            return None
        path.append(next(iter(next_atoms)))
    return tuple(path)


def _acid_fragment_name(fragment, Chem):
    """Name one unmodified, linear dicarboxylic acid component."""
    source_atom_count = fragment.GetNumAtoms()
    # The component partition cuts each original ester O away with the
    # alcoholic component. Reconstruct ordinary OH caps on a temporary copy
    # for acid-parent selection; these synthetic atoms are never assigned to
    # the input's complete-role map.
    capped = Chem.RWMol(fragment)
    carbonyl_centers = []
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        doubles = [n for n in atom.GetNeighbors()
                   if n.GetAtomicNum() == 8
                   and fragment.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        singles = [n for n in atom.GetNeighbors()
                   if n.GetAtomicNum() == 8
                   and fragment.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(doubles) == 1 and not singles:
            carbonyl_centers.append(atom.GetIdx())
    if len(carbonyl_centers) != 2:
        return None
    for center in carbonyl_centers:
        oxygen = capped.AddAtom(Chem.Atom(8))
        capped.AddBond(center, oxygen, Chem.BondType.SINGLE)
    fragment = capped.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except (ValueError, RuntimeError):
        return None
    from .extended_inventory import selection_inventory
    from .complete_functional_parent import functional_parent_candidates
    from .acid_derivative_methods import _anion

    inventory = selection_inventory(fragment)
    acids = tuple(g for g in inventory.groups if g.kind == "acid")
    if (not inventory.supported or len(acids) != 2
            or len(inventory.groups) != 2
            or frozenset(inventory.principal) != frozenset(acids)):
        return None
    if any(a.GetAtomicNum() not in {6, 8} or a.GetIsotope()
           or int(a.GetChiralTag()) for a in fragment.GetAtoms()):
        return None
    carbon_atoms = {a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 6}
    acid_centers = {g.center for g in acids}
    skeleton = carbon_atoms - acid_centers
    path = _carbon_path(fragment, skeleton)
    if path is None or not skeleton:
        return None
    # Both carboxyl groups must terminate the same simple carbon path.
    hosts = []
    for group in acids:
        neighbors = [n.GetIdx() for n in fragment.GetAtomWithIdx(group.center).GetNeighbors()
                     if n.GetAtomicNum() == 6 and n.GetIdx() in skeleton]
        if len(neighbors) != 1:
            return None
        hosts.append(neighbors[0])
    if set(hosts) != {path[0], path[-1]}:
        return None
    # Four oxygens only: two carbonyl O and two hydroxyl O.
    if sum(a.GetAtomicNum() == 8 for a in fragment.GetAtoms()) != 4:
        return None
    candidates = functional_parent_candidates(fragment, Chem, include_single=True,
                                               principal=acids)
    if not candidates:
        return None
    # For this unbranched, end-functionalized carbon path, the named parent
    # and its two suffix positions are chemically invariant under reversal.
    names = {candidate.name for candidate in candidates
             if candidate.principal_centers == frozenset(g.center for g in acids)}
    if len(names) != 1:
        return None
    acid_name = next(iter(names))
    anion = _anion(acid_name)
    if not anion or not re.fullmatch(r"[a-z]+ate", anion):
        return None
    candidate = next(row for row in candidates
                     if row.principal_centers == frozenset(g.center for g in acids)
                     and row.name == acid_name)
    original_locants = tuple((idx, str(locant)) for idx, locant in candidate.atom_locants
                             if idx < source_atom_count)
    if not original_locants:
        return None
    return acid_name, anion, original_locants


def _alcohol_fragment(fragment, original_atoms, original_mol, Chem, bridge_o, root,
                      degree):
    carbon_atoms = {a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 6}
    if (not carbon_atoms or any(a.GetAtomicNum() not in {6, 8} or a.GetIsotope()
                                or int(a.GetChiralTag()) for a in fragment.GetAtoms())):
        return None
    path = _carbon_path(fragment, carbon_atoms)
    if path is None:
        return None
    oxygens = [a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 8]
    if len(oxygens) != degree:
        return None
    mapped_oxygen = original_atoms.index(bridge_o)
    if mapped_oxygen not in oxygens:
        return None
    attachment_atoms = []
    for oxygen in oxygens:
        neighbors = [n.GetIdx() for n in fragment.GetAtomWithIdx(oxygen).GetNeighbors()
                     if n.GetAtomicNum() == 6]
        if len(neighbors) != 1 or fragment.GetBondBetweenAtoms(oxygen, neighbors[0]).GetBondTypeAsDouble() != 1:
            return None
        attachment_atoms.append(neighbors[0])
    if degree == 1:
        local_root = attachment_atoms[0]
        name, atoms = mapped_branch(fragment, Chem, local_root, mapped_oxygen)
        if not name or not re.fullmatch(r"[a-z]+yl", name):
            return None
        # mapped_branch excludes the ester O, which belongs to this alcohol
        # role as the original functional-class bridge atom.
        if set(atoms) | {mapped_oxygen} != set(range(fragment.GetNumAtoms())):
            return None
        return {"degree": 1, "name": name,
                "identity": Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True),
                "atoms": frozenset(original_atoms), "attachment": attachment_atoms[0],
                "atom_locants": (), "attachment_locants": (),
                "attachment_original": original_atoms[attachment_atoms[0]]}

    if degree != 2:
        return None
    if len(path) not in ROOTS or len(set(attachment_atoms)) != 2:
        return None
    locants_by_orientation = []
    for order in (path, tuple(reversed(path))):
        locants = tuple(sorted(order.index(site) + 1 for site in attachment_atoms))
        locants_by_orientation.append((locants, order))
    locants, order = min(locants_by_orientation, key=lambda row: row[0])
    if len(set(locants)) != 2:
        return None
    parent = ROOTS[len(path)] + "ane"
    name = f"{parent}-{','.join(map(str, locants))}-diyl"
    if set(oxygens) | carbon_atoms != set(range(fragment.GetNumAtoms())):
        return None
    atom_locants = tuple((original_atoms[local], str(position))
                         for position, local in enumerate(order, 1))
    attachment_locants = tuple((original_atoms[oxygen],
                                str(order.index(host) + 1))
                               for oxygen in oxygens
                               for host in [next(n.GetIdx() for n in fragment.GetAtomWithIdx(oxygen).GetNeighbors()
                                                 if n.GetAtomicNum() == 6)])
    return {"degree": 2, "name": name,
            "identity": Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True),
            "atoms": frozenset(original_atoms), "attachment_locs": locants,
            "atom_locants": atom_locants, "attachment_locants": attachment_locants}


def polyester_assembly_decision(mol, Chem):
    """Return the bounded P-65.6.3.3.4.1 functional-class decision or ``None``."""
    if (mol.GetNumHeavyAtoms() > 128 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()):
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           or int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    esters = tuple(g for g in inventory.groups if g.kind == "ester")
    if (not inventory.supported or len(esters) != 4
            or len(inventory.groups) != 4 or len(inventory.principal) != 4
            or any(g.kind != "ester" for g in inventory.principal)):
        return None

    # Disconnect each acyl--organyl bond simultaneously. The bridge oxygens
    # remain attached to their original alcohol-side atoms; the acid-side
    # carbonyls become ordinary hydroxy-capped acid sites after sanitization.
    cut = Chem.RWMol(mol)
    for group in esters:
        if cut.GetBondBetweenAtoms(group.center, group.ligand) is None:
            return None
        cut.RemoveBond(group.center, group.ligand)
    cut = cut.GetMol()
    try:
        Chem.SanitizeMol(cut)
    except (ValueError, RuntimeError):
        return None
    fragments = _fragments(cut, Chem)
    atom_to_component = {}
    for ci, (_, mapping) in enumerate(fragments):
        for atom_idx in mapping:
            atom_to_component[atom_idx] = ci
    acid_components, alcohol_components = set(), set()
    ester_edges = []
    for group in esters:
        acid_id = atom_to_component[group.center]
        alcohol_id = atom_to_component[group.ligand]
        if acid_id == alcohol_id:
            return None
        acid_components.add(acid_id)
        alcohol_components.add(alcohol_id)
        ester_edges.append((acid_id, alcohol_id, group))
    if len(acid_components) != 2 or len(alcohol_components) != 3:
        return None

    # The source-backed incidence graph is a path A--O--A with one terminal
    # monovalent alcoholic component at each acid residue.
    acid_degrees = {ci: 0 for ci in acid_components}
    alcohol_degrees = {ci: 0 for ci in alcohol_components}
    for acid_id, alcohol_id, _ in ester_edges:
        acid_degrees[acid_id] += 1
        alcohol_degrees[alcohol_id] += 1
    if set(acid_degrees.values()) != {2} or sorted(alcohol_degrees.values()) != [1, 1, 2]:
        return None
    central_id = next(ci for ci, degree in alcohol_degrees.items() if degree == 2)
    if {acid_id for acid_id, alcohol_id, _ in ester_edges if alcohol_id == central_id} != acid_components:
        return None
    for acid_id in acid_components:
        if sum(edge[0] == acid_id and edge[1] == central_id for edge in ester_edges) != 1:
            return None
        if sum(edge[0] == acid_id and edge[1] != central_id for edge in ester_edges) != 1:
            return None

    named_acids = {}
    for ci in acid_components:
        fragment, mapping = fragments[ci]
        named = _acid_fragment_name(fragment, Chem)
        if named is None:
            return None
        named_acids[ci] = (named, frozenset(mapping),
                           tuple((mapping[local], locant)
                                 for local, locant in named[2]))
    if len({(row[0][0], row[0][1],
             Chem.MolToSmiles(fragments[ci][0], canonical=True, isomericSmiles=True))
            for ci, row in named_acids.items()}) != 1:
        return None
    acid_name, anion, _ = next(iter(named_acids.values()))[0]

    alcohol_rows = {}
    for ci in alcohol_components:
        fragment, mapping = fragments[ci]
        attached_edges = [(acid_id, group) for acid_id, alcohol_id, group in ester_edges
                          if alcohol_id == ci]
        bridge_o = attached_edges[0][1].ligand
        oxygen_local = mapping.index(bridge_o)
        roots = [n.GetIdx() for n in fragment.GetAtomWithIdx(oxygen_local).GetNeighbors()
                 if n.GetAtomicNum() == 6]
        if len(roots) != 1:
            return None
        degree = alcohol_degrees[ci]
        row = _alcohol_fragment(fragment, mapping, mol, Chem, bridge_o, roots[0], degree)
        if row is None:
            return None
        alcohol_rows[ci] = row
    terminal_ids = [ci for ci in alcohol_components if ci != central_id]
    central = alcohol_rows[central_id]
    if len(terminal_ids) != 2:
        return None
    terminal_partner = {}
    for terminal_id in terminal_ids:
        edges = [(acid_id, group) for acid_id, alcohol_id, group in ester_edges
                 if alcohol_id == terminal_id]
        if len(edges) != 1:
            return None
        terminal_partner[terminal_id] = edges[0]
    def terminal_site_locant(terminal_id):
        _, group = terminal_partner[terminal_id]
        locants = dict(central["attachment_locants"])
        return int(locants.get(group.ligand, "0"))

    terminal_ids.sort(key=lambda ci: (
        prefix_alphabetization_key(alcohol_rows[ci]["name"]),
        terminal_site_locant(ci),
    ))
    terminal_rows = [alcohol_rows[ci] for ci in terminal_ids]
    # Distinct rooted graphs that collapse to the same citation are not
    # silently merged into a multiplier.
    if (terminal_rows[0]["identity"] != terminal_rows[1]["identity"]
            and terminal_rows[0]["name"] == terminal_rows[1]["name"]):
        return None
    terminal_species = {}
    for row in terminal_rows:
        key = row["identity"]
        terminal_species.setdefault(key, {"name": row["name"], "count": 0})
        terminal_species[key]["count"] += 1
    terminal_words = []
    for species in sorted(terminal_species.values(),
                          key=lambda item: prefix_alphabetization_key(item["name"])):
        name = species["name"]
        count = species["count"]
        terminal_words.append(name if count == 1 else MULT[count] + name)

    name = " ".join(terminal_words + [central["name"], MULT[2] + anion])
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    roles = []
    for number, ci in enumerate(terminal_ids, 1):
        acid_id, _ = terminal_partner[ci]
        roles.append((f"terminal-alcohol:{number}", alcohol_rows[ci]["atoms"],
                      alcohol_rows[ci]["atom_locants"]))
        roles.append((f"acid-for-terminal-alcohol:{number}",
                      named_acids[acid_id][1], named_acids[acid_id][2]))
    roles.append(("central-linker", central["atoms"], central["atom_locants"]))
    if set().union(*(set(row[1]) for row in roles)) != set(heavy):
        return None
    if any(set(left[1]) & set(right[1]) for i, left in enumerate(roles)
           for right in roles[i + 1:]):
        return None
    centers = frozenset(group.center for group in esters)
    return MethodDecision(name, "functional-class", centers, heavy,
                          "P-65.6.3.3.4.1", (), component_roles=tuple(roles))
