"""Mapped linear dianhydrides with a central alkanedioic acid residue.

This is the bounded P-65.7.6.1 construction where two terminal monocarboxylic
acid residues are attached through anhydride oxygen atoms to opposite ends of
one unbranched saturated alkanedioic acid residue.
"""

from .molecule_iteration import iter_atoms, iter_bonds
from .acid_derivative_methods import _acid_component
from .component_mapping import MappedComponent, decorate_component
from .extended_inventory import selection_inventory
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision


def _carbonyl_oxygens(mol, center):
    return tuple(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(center).GetNeighbors()
                 if neighbor.GetAtomicNum() == 8
                 and mol.GetBondBetweenAtoms(center, neighbor.GetIdx()).GetBondTypeAsDouble() == 2)


def _anhydride_bridges(mol):
    result = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 8 or atom.GetDegree() != 2:
            continue
        neighbors = tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        if (len(neighbors) == 2 and all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
                                        and len(_carbonyl_oxygens(mol, idx)) == 1
                                        for idx in neighbors)
                and all(mol.GetBondBetweenAtoms(atom.GetIdx(), idx).GetBondTypeAsDouble() == 1
                        for idx in neighbors)):
            result.append((atom.GetIdx(), neighbors))
    return tuple(result)


def _components_without(mol, excluded):
    excluded = set(excluded)
    remaining = set(range(mol.GetNumAtoms())) - excluded
    components = []
    while remaining:
        start = min(remaining)
        todo = [start]
        component = set()
        while todo:
            idx = todo.pop()
            if idx in component or idx in excluded:
                continue
            component.add(idx)
            remaining.discard(idx)
            todo.extend(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if neighbor.GetIdx() not in excluded and neighbor.GetIdx() not in component)
        components.append(frozenset(component))
    return tuple(components)


def _central_chain_ok(mol, atoms, centers):
    """Require an unbranched carbon path with the two acid centers at its ends."""
    carbon_atoms = {idx for idx in atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6}
    if len(centers) != 2 or not centers <= carbon_atoms:
        return False
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() not in {1, 6, 8} for idx in atoms):
        return False
    double_oxygens = {oxygen for center in centers for oxygen in _carbonyl_oxygens(mol, center)}
    if len(double_oxygens) != 2:
        return False
    heavy = {idx for idx in atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    if heavy != carbon_atoms | double_oxygens:
        return False
    adjacency = {idx: set() for idx in carbon_atoms}
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in carbon_atoms or right not in carbon_atoms:
            continue
        if bond.GetBondTypeAsDouble() != 1:
            return False
        adjacency[left].add(right)
        adjacency[right].add(left)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return False
    if any(len(adjacency[idx]) != 1 for idx in centers):
        return False
    if any(not neighbors and idx not in centers for idx, neighbors in adjacency.items()):
        return False
    seen, todo = set(), [next(iter(centers))]
    while todo:
        idx = todo.pop()
        if idx in seen:
            continue
        seen.add(idx)
        todo.extend(adjacency[idx] - seen)
    return seen == carbon_atoms and all(len(adjacency[idx]) == 2 for idx in carbon_atoms - centers)


def _mapped_diacid(mol, Chem, atoms, centers):
    """Cap a central dicarboxy component twice, select its acid parent, remap."""
    order = tuple(sorted(atoms))
    local = {original: index for index, original in enumerate(order)}
    graph = Chem.RWMol()
    for idx in order:
        graph.AddAtom(Chem.Atom(mol.GetAtomWithIdx(idx)))
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in local and right in local:
            graph.AddBond(local[left], local[right], bond.GetBondType())
    caps = []
    for center in sorted(centers):
        cap = graph.AddAtom(Chem.Atom(8))
        graph.AddBond(local[center], cap, Chem.BondType.SINGLE)
        caps.append(cap)
    acid = graph.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None

    parent_graph = Chem.Mol(acid)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.GetAtomicNum() == 8:
            atom.SetIsotope(0)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    inventory = selection_inventory(parent_graph)
    if len(inventory.principal) != 2 or any(group.kind != "acid" for group in inventory.principal):
        return None
    from .cross_route_selection import cross_route_selection
    selection = cross_route_selection(parent_graph, Chem)
    if selection is None:
        return None
    construction = selection.selected.construction
    if construction is None or not construction.parent.endswith("dioic acid"):
        return None
    cap_set = set(caps)
    mapped_atoms = {order[index] for index in construction.atoms if index < len(order)}
    mapped_locants = {order[index]: locant for index, locant in construction.atom_locants
                      if index < len(order) and index not in cap_set}
    for index, locant in selection.selected.atom_locants:
        if index < len(order) and index not in cap_set:
            mapped_locants.setdefault(order[index], locant)
            mapped_atoms.add(order[index])
    expected = {idx for idx in atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    if mapped_atoms != expected or set(mapped_locants) != mapped_atoms:
        return None
    translated = MappedComponent(
        construction.parent,
        tuple(mapped_locants.items()),
        frozenset(mapped_atoms),
        construction.prefixes,
        frozenset(order[index] for index in construction.embedded_atoms if index < len(order)),
        construction.hydro_prefix,
        construction.systematic_parent,
        construction.nitrogen_site_locants,
        construction.isotope_unlocanted_elements,
        construction.prefix_mode,
        frozenset(order[index] for index in construction.local_isotope_atoms
                  if index < len(order) and order[index] in mapped_atoms),
        construction.omit_methyl_prefix_locant,
        construction.isotope_descriptor_parts,
    )
    name = decorate_component(mol, Chem, translated)
    if not name:
        return None
    return name, mapped_atoms, tuple(mapped_locants.items())


def _acid_group(name):
    return name[:-5] if name.endswith(" acid") else ""


def poly_anhydride_decision(mol, Chem):
    """Return a mapped P-65.7.6.1 linear dianhydride decision, or None."""
    if (mol.GetNumHeavyAtoms() > 96 or mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                   or atom.GetAtomicNum() not in {6, 8, 9, 17, 35, 53}
                   or atom.GetIsotope() and atom.GetAtomicNum() != 6
                   or atom.GetAtomicNum() != 6 and int(atom.GetChiralTag())
                   for atom in iter_atoms(mol))
            or any(int(bond.GetStereo()) for bond in iter_bonds(mol))):
        return None
    bridges = _anhydride_bridges(mol)
    if len(bridges) != 2:
        return None
    bridge_atoms = frozenset(bridge for bridge, _ in bridges)
    carbonyl_centers = {atom.GetIdx() for atom in iter_atoms(mol)
                        if atom.GetAtomicNum() == 6 and len(_carbonyl_oxygens(mol, atom.GetIdx())) == 1}
    if len(carbonyl_centers) != 4:
        return None
    components = _components_without(mol, bridge_atoms)
    if len(components) != 3:
        return None
    component_centers = {component: carbonyl_centers & component for component in components}
    central_candidates = [component for component in components if len(component_centers[component]) == 2]
    terminal_components = [component for component in components if len(component_centers[component]) == 1]
    if len(central_candidates) != 1 or len(terminal_components) != 2:
        return None
    central = central_candidates[0]
    central_centers = component_centers[central]
    if not _central_chain_ok(mol, central, central_centers):
        return None
    terminal_centers = {next(iter(component_centers[component])) for component in terminal_components}
    if any(sum(center in endpoints for center in central_centers) != 1
           or sum(center in endpoints for center in terminal_centers) != 1
           for _, endpoints in bridges):
        return None

    terminal_parts = []
    for component in terminal_components:
        center = next(iter(component_centers[component]))
        bridge_matches = [bridge for bridge, endpoints in bridges if center in endpoints]
        if len(bridge_matches) != 1:
            return None
        part = _acid_component(mol, Chem, center, bridge_matches[0])
        if part is None or part.atoms != component or not part.name.endswith(" acid"):
            return None
        terminal_parts.append((part, center, bridge_matches[0]))

    central_part = _mapped_diacid(mol, Chem, central, central_centers)
    if central_part is None:
        return None
    central_name, central_atoms, central_locants = central_part
    central_group = _acid_group(central_name)
    end_keys = [prefix_alphabetization_key(_acid_group(row[0].name)) for row in terminal_parts]
    end_names_unsorted = [_acid_group(row[0].name) for row in terminal_parts]
    if end_keys[0] == end_keys[1] and end_names_unsorted[0] != end_names_unsorted[1]:
        return None
    end_rows = [row for _, row in sorted(zip(end_keys, terminal_parts), key=lambda pair: pair[0])]
    end_names = [_acid_group(row[0].name) for row in end_rows]
    if any(not name for name in end_names) or not central_group:
        return None
    end_prefix = "di" + end_names[0] if end_names[0] == end_names[1] else end_names[0]
    ordered_names = [end_prefix, central_group]
    if end_names[0] != end_names[1]:
        ordered_names.append(end_names[1])
    name = " ".join(ordered_names + ["dianhydride"])

    atoms = frozenset(atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1)
    terminal_atom_union = terminal_parts[0][0].atoms | terminal_parts[1][0].atoms
    if (terminal_atom_union & central_atoms or terminal_atom_union & bridge_atoms
            or central_atoms & bridge_atoms
            or terminal_atom_union | central_atoms | bridge_atoms != atoms):
        return None
    centers = frozenset(carbonyl_centers)
    roles = [
        ("terminal-acid:1", end_rows[0][0].atoms, end_rows[0][0].atom_locants),
        ("central-diacid", central_atoms, central_locants),
        ("terminal-acid:2", end_rows[1][0].atoms, end_rows[1][0].atom_locants),
    ]
    roles.extend(("anhydride-oxygen", frozenset({bridge}), ()) for bridge in sorted(bridge_atoms))
    locants = tuple(end_rows[0][0].atom_locants) + tuple(central_locants) + tuple(end_rows[1][0].atom_locants)
    return MethodDecision(name, "functional-class", centers, atoms, "P-65.7.6.1",
                          locants, component_roles=tuple(roles))
