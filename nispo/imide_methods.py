"""Mapped, bounded names for acyclic imides (Blue Book P-66.2.1).

The producer follows the Blue Book instruction to name an acyclic imide as
an N-acyl derivative of its senior primary amide (P-66.1.2.1). It keeps the
two acid-derived components separate through selection and rendering so atom
ownership, isotope locants, and CIP descriptors remain attached to the input
graph. Cyclic imide nitrogens are left to the heterocyclic pseudoketone route.
"""

from .molecule_iteration import iter_atoms
from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
from .naming_methods import MethodDecision


RULE = "P-41/P-44/P-45/P-66.1.2.1/P-66.2.1/P-14.5/P-16.2.4/P-16.5.1"


def _bond_order(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond.GetBondTypeAsDouble() if bond else 0


def _amide_parent(acid_name):
    """Return a source-backed primary amide form for ordinary carboxylic acids."""
    for acid, amide in (("formic acid", "formamide"),
                        ("acetic acid", "acetamide"),
                        ("benzoic acid", "benzamide")):
        if acid_name.endswith(acid):
            return acid_name[:-len(acid)] + amide
    if acid_name.endswith("carboxylic acid"):
        return acid_name[:-len("carboxylic acid")] + "carboxamide"
    if acid_name.endswith("oic acid"):
        return acid_name[:-len("oic acid")] + "amide"
    return ""


def _acid_parent_facts(mol, Chem, center, leaving):
    """Get the acid-component parent facts used for P-44 arm comparison.

    This mirrors the acid derivative's temporary acid-cap operation, then
    asks the shared cross-route selector for its mapped parent facts. It does
    not rank or compare rendered names.
    """
    from .branch_rules import collect_subtree
    from .cross_route_selection import cross_route_selection

    atoms = frozenset(collect_subtree(mol, center, leaving))
    if not atoms or leaving in atoms:
        return None
    order = sorted(atoms)
    graph = Chem.RWMol(mol)
    for idx in reversed(range(mol.GetNumAtoms())):
        if idx not in atoms:
            graph.RemoveAtom(idx)
    cap = Chem.Atom(8)
    oxygen = graph.AddAtom(cap)
    graph.AddBond(order.index(center), oxygen, Chem.BondType.SINGLE)
    acid = graph.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None
    # P-44 comparison first uses the parent structures without the locally
    # labelled acid suffix oxygen; carbon isotope labels remain part of the
    # mapped parent, as required for component identity and later decoration.
    parent_graph = Chem.Mol(acid)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    from .acid_derivative_methods import _capped_acid_selection
    selection = _capped_acid_selection(
        Chem, parent_graph.ToBinary(Chem.PropertyPickleOptions.AllProps),
        cross_route_selection)
    if selection is None or selection.selected.construction is None:
        return None
    construction = selection.selected.construction
    if not construction.parent.endswith(" acid"):
        return None
    return selection.selected.parent_selection


def _site(mol, nitrogen):
    atom = mol.GetAtomWithIdx(nitrogen)
    if (atom.GetAtomicNum() != 7 or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.IsInRing()
            or atom.GetIsAromatic() or atom.GetIsotope()
            or int(atom.GetChiralTag())):
        return None
    heavy = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
             if neighbor.GetAtomicNum() > 1]
    if len(heavy) not in {2, 3}:
        return None
    acyl = []
    for idx in heavy:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 or _bond_order(mol, nitrogen, idx) != 1:
            continue
        oxygens = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                   if neighbor.GetAtomicNum() == 8 and _bond_order(mol, idx, neighbor.GetIdx()) == 2]
        if len(oxygens) == 1:
            acyl.append((idx, oxygens[0]))
    if len(acyl) != 2 or any(idx not in heavy for idx, _ in acyl):
        return None
    remaining = [idx for idx in heavy if idx not in {center for center, _ in acyl}]
    if len(remaining) != len(heavy) - 2:
        return None
    if remaining and (len(remaining) != 1
                      or mol.GetAtomWithIdx(remaining[0]).GetAtomicNum() != 6
                      or _bond_order(mol, nitrogen, remaining[0]) != 1):
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    return tuple(acyl), (remaining[0] if remaining else None)


def imide_decision(mol, Chem):
    """Return a mapped N-acyl-primary-amide decision for a bounded acyclic imide."""
    from .acid_derivative_methods import _acid_component, _acyl
    from .component_mapping import mapped_branch, decorate_component
    from dataclasses import replace
    from .parent_selection import compare_parents

    if mol.GetStereoGroups():
        return None
    # P-66.1.2.1's bounded construction is proved here for carbon-isotope
    # and tetrahedral CIP decoration through the acid component mapper. Oxygen
    # and nitrogen isotope scope on amide/N-acyl endings is not covered by the
    # cited imide examples, so it remains explicitly deferred.
    if any(atom.GetIsotope() and atom.GetAtomicNum() != 6 for atom in iter_atoms(mol)):
        return None
    nitrogens = [atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() == 7]
    sites = [(nitrogen, _site(mol, nitrogen)) for nitrogen in nitrogens]
    sites = [(nitrogen, site) for nitrogen, site in sites if site is not None]
    if len(sites) != 1:
        return None
    nitrogen, (acyl_centers, organyl_root) = sites[0]
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if (any(group.kind == "amide" and group.center not in {center for center, _ in acyl_centers}
            for group in inventory.groups)
            or any(group.kind != "amide" and group.seniority < (11, 0)
                   for group in inventory.groups)):
        # A more senior characteristic group can require the amides in
        # prefix mode; that complete polyfunctional construction is outside
        # this N-acyl suffix producer.
        return None
    heavy_atoms = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                            if atom.GetAtomicNum() > 1)
    components = []
    for center, carbonyl_oxygen in acyl_centers:
        part = _acid_component(mol, Chem, center, nitrogen)
        if part is None or center not in part.atoms or carbonyl_oxygen not in part.atoms:
            return None
        if part.atoms & {nitrogen}:
            return None
        components.append((center, carbonyl_oxygen, part))
    (left_center, _, left), (right_center, _, right) = components
    if left.atoms & right.atoms:
        return None

    # Compare the corresponding acid-derived parent structures according to
    # P-41/P-44. A P-44 tie is usable only for graph-identical acid components;
    # otherwise the choice of base arm remains unresolved here.
    left_facts = _acid_parent_facts(mol, Chem, left_center, nitrogen)
    right_facts = _acid_parent_facts(mol, Chem, right_center, nitrogen)
    comparison = compare_parents(left_facts, right_facts)
    if comparison.preference == -1:
        parent_arm, prefix_arm = left, right
    elif comparison.preference == 1:
        parent_arm, prefix_arm = right, left
    elif comparison.preference == 0 and left.identity == right.identity:
        parent_arm, prefix_arm = left, right
    else:
        return None

    if parent_arm.construction is None:
        return None
    parent_component = parent_arm.construction
    parent = _amide_parent(parent_component.parent)
    prefix = _acyl(prefix_arm.name)
    if not parent or not prefix:
        return None
    n_prefixes = [("N", prefix)]
    owned = set(parent_arm.atoms) | set(prefix_arm.atoms) | {nitrogen}
    roles = [("primary-amide-parent", parent_arm.atoms, parent_arm.atom_locants),
             ("N-acyl", prefix_arm.atoms, prefix_arm.atom_locants),
             ("imide-nitrogen", frozenset({nitrogen}), ())]
    if organyl_root is not None:
        organyl, organyl_atoms = mapped_branch(mol, Chem, organyl_root, nitrogen)
        if not organyl or organyl_atoms & owned:
            return None
        n_prefixes.append(("N", organyl))
        owned.update(organyl_atoms)
        roles.append(("N-organyl", organyl_atoms, ()))
    if owned != set(heavy_atoms):
        return None

    # P-14.5 order is applied by the shared formatter; its enclosure rules
    # produce the P-16.9 parentheses needed for complex acyl substituents.
    # The parent acid is a typed undecorated unit. Add N-substitution before
    # reading original-graph stereo and skeletal isotope descriptors, so a
    # parent's R/S descriptor remains at the front of the complete amide.
    mapping = dict(parent_component.atom_locants)
    mapping[nitrogen] = 'N'
    component = replace(
        parent_component, parent=parent,
        atom_locants=tuple(mapping.items()),
        atoms=frozenset(mapping),
        prefixes=parent_component.prefixes + tuple(n_prefixes),
        embedded_atoms=frozenset(heavy_atoms - mapping.keys()),
        nitrogen_site_locants=True,
    )
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    atom_locants = parent_arm.atom_locants + prefix_arm.atom_locants + ((nitrogen, "N"),)
    rule = RULE
    return MethodDecision(name, "substitutive", frozenset({left_center, right_center}),
                          heavy_atoms, rule, atom_locants,
                          component_roles=tuple(roles))
