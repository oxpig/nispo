"""Mapped functional-class names for a bounded ordinary carbamate ester."""

from .molecule_iteration import iter_atoms
from .component_mapping import (MappedComponent, chemical_inventory,
                                decorate_component, mapped_branch)
from .naming_methods import MethodDecision


_RULE = ("P-65.2.1.1/P-65.6.3.2.1/P-65.6.3.3.1/"
         "P-82.2.1/P-82.2.4/P-82.2.5/P-82.6.1.2")


def mapped_carbamate_amino_prefix(mol, Chem, root, host, scope):
    """An ordinary N-attached alkoxycarbonylamino arm on the original graph."""
    atom = mol.GetAtomWithIdx(root)
    if (atom.IsInRing() or atom.GetAtomicNum() != 7 or atom.GetDegree() != 2
            or atom.GetTotalNumHs(includeNeighbors=True) != 1
            or atom.GetFormalCharge() or atom.GetIsotope()
            or mol.GetBondBetweenAtoms(root, host) is None
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
        return None
    carbon = next(a for a in atom.GetNeighbors() if a.GetIdx() != host)
    if carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3:
        return None
    oxygens = {b.GetBondTypeAsDouble(): b.GetOtherAtom(carbon) for b in carbon.GetBonds()
               if b.GetOtherAtom(carbon).GetAtomicNum() == 8}
    if (oxygens.keys() != {1, 2} or oxygens[2].GetDegree() != 1
            or oxygens[1].GetDegree() != 2
            or any(n.GetAtomicNum() != 6 for n in oxygens[1].GetNeighbors())
            or any(a.GetIsotope() or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() for a in (carbon, *oxygens.values()))):
        return None
    bridge = oxygens[1].GetIdx()
    from .branch.contracted_alkoxy import mapped_contracted_alkoxy

    alkoxy = mapped_contracted_alkoxy(mol, Chem, bridge, carbon.GetIdx())
    oxy, owned = alkoxy if alkoxy is not None else mapped_branch(mol, Chem, bridge, carbon.GetIdx())
    core = {root, carbon.GetIdx(), oxygens[2].GetIdx()}
    if not oxy or not owned or owned & core or owned | core != scope:
        return None
    from .component_mapping import mapped_branch_ring_atoms, _record_branch_ring_atoms
    from .formatting import enclose_name_unit
    name = enclose_name_unit(oxy + "carbonyl") + "amino"
    _record_branch_ring_atoms(mol, Chem, root, host, name, scope,
                              mapped_branch_ring_atoms(mol, Chem, bridge, carbon.GetIdx(), oxy, owned))
    return name, scope


def _carbamate_core(mol, Chem):
    """Return the only exact C(=O)(O)(N) core, if the graph has one."""
    cores = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() != 0:
            continue
        cidx = carbon.GetIdx()
        double_o, single_o, single_n = [], [], []
        valid = True
        for neighbor in carbon.GetNeighbors():
            idx = neighbor.GetIdx()
            order = mol.GetBondBetweenAtoms(cidx, idx).GetBondTypeAsDouble()
            if neighbor.GetAtomicNum() == 8 and order == 2:
                double_o.append(idx)
            elif neighbor.GetAtomicNum() == 8 and order == 1:
                single_o.append(idx)
            elif neighbor.GetAtomicNum() == 7 and order == 1:
                single_n.append(idx)
            else:
                valid = False
                break
        if valid and len(double_o) == len(single_o) == len(single_n) == 1:
            cores.append((cidx, double_o[0], single_o[0], single_n[0]))
    return cores[0] if len(cores) == 1 else None


def _carbon_organyl(mol, Chem, root, blocked):
    """Name one carbon-only organyl branch and prove its original atom set."""
    name, owned = mapped_branch(mol, Chem, root, blocked)
    if not name or not owned or root not in owned:
        return "", frozenset()
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned):
        return "", frozenset()
    return name, frozenset(owned)


def _unbranched_saturated_carbon_path(mol, atoms, root):
    """Prove one saturated carbon path rooted at an end on the original graph."""
    atoms = frozenset(atoms)
    if not atoms or root not in atoms:
        return False
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetIsotope()
                or int(atom.GetChiralTag())):
            return False
        neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in atoms]
        if len(neighbors) > 2 or idx == root and len(neighbors) > 1:
            return False
        if any(mol.GetBondBetweenAtoms(idx, neighbor).GetBondTypeAsDouble() != 1
               for neighbor in neighbors):
            return False
    # The enclosing scope is connected, but the carbon-only induced graph may
    # not be: explicitly prove a single path instead of relying on atom count.
    seen, todo = set(), [root]
    while todo:
        idx = todo.pop()
        if idx in seen:
            continue
        seen.add(idx)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if n.GetIdx() in atoms and n.GetIdx() not in seen)
    return seen == set(atoms)


def _haloalkyl_organyl(mol, Chem, root, blocked):
    """Prove a terminally halo-substituted, one-cut O-organyl path.

    Only the original neutral carbon path and terminal halogens are admitted.
    The mapped branch writer must own exactly that component. Its cached text
    is accepted only after this topology and its rooted orientation are proved.
    """
    from .branch_rules import collect_subtree

    scope = frozenset(collect_subtree(mol, root, blocked))
    carbons = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    halogens = frozenset(i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() in {9, 17, 35, 53})
    if (not halogens or not 1 <= len(carbons) <= 6
            or carbons | halogens != scope
            or not _unbranched_saturated_carbon_path(mol, carbons, root)):
        return "", frozenset()
    for idx in halogens:
        atom = mol.GetAtomWithIdx(idx)
        neighbors = tuple(atom.GetNeighbors())
        if (atom.GetIsotope() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or len(neighbors) != 1 or neighbors[0].GetIdx() not in carbons
                or mol.GetBondBetweenAtoms(idx, neighbors[0].GetIdx()).GetBondTypeAsDouble() != 1):
            return "", frozenset()
        attached = neighbors[0].GetIdx()
        if attached == root and len(carbons) > 1:
            return "", frozenset()
        if sum(n.GetIdx() in carbons for n in neighbors[0].GetNeighbors()) > 1:
            return "", frozenset()
    name, owned = mapped_branch(mol, Chem, root, blocked)
    return (name, scope) if name and frozenset(owned) == scope else ("", frozenset())


def _proved_organyl(mol, Chem, root, blocked, *, allow_n_oxygen=False):
    """Admit an independently owned carbon, mapped ring, or N-oxy arm."""
    from .branch_rules import collect_subtree
    from .formatting import enclose_name_unit
    scope = frozenset(collect_subtree(mol, root, blocked))
    atom = mol.GetAtomWithIdx(root)
    if not scope or root not in scope:
        return "", frozenset()
    carbon = _carbon_organyl(mol, Chem, root, blocked)
    if carbon[0] and carbon[1] == scope:
        return carbon
    if allow_n_oxygen and atom.GetAtomicNum() == 8:
        if (atom.IsInRing() or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetTotalNumHs(includeNeighbors=True)
                or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
            return "", frozenset()
        children = [a for a in atom.GetNeighbors()
                    if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
        if (len(children) != 1 or children[0].GetAtomicNum() != 6
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                       for i in scope - {root})):
            return "", frozenset()
        child_name, child_atoms = _carbon_organyl(mol, Chem, children[0].GetIdx(), root)
        if not child_name or child_atoms | {root} != scope:
            return "", frozenset()
        name, owned = mapped_branch(mol, Chem, root, blocked)
        return (name, owned) if name and owned == scope else ("", frozenset())
    if atom.GetAtomicNum() != 6:
        return "", frozenset()
    from .annular_carbamate_carboxylate_methods import _proved_ring_branch
    if atom.IsInRing():
        return _proved_ring_branch(mol, Chem, root, blocked)
    if allow_n_oxygen:
        from .internally_rooted_carbamate import internally_rooted_carbamate_branch
        internal = internally_rooted_carbamate_branch(mol, Chem, root, blocked)
        if internal is not None:
            return internal
    # A single N-CH2-ring unit can own a complete, independently numbered
    # cyclic component. Avoid inferring larger carbon-chain parent choices.
    children = [a for a in atom.GetNeighbors()
                if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
    if (atom.GetTotalNumHs(includeNeighbors=True) != 2 or len(children) != 1
            or not children[0].IsInRing()
            or mol.GetBondBetweenAtoms(root, children[0].GetIdx()).GetBondTypeAsDouble() != 1):
        return "", frozenset()
    ring_name, ring_atoms = _proved_ring_branch(mol, Chem, children[0].GetIdx(), root)
    if not ring_name or ring_atoms | {root} != scope:
        return "", frozenset()
    return enclose_name_unit(ring_name) + "methyl", scope


def carbamate_decision(mol, Chem, *, allow_haloalkyl=False):
    """Return a fully owned ordinary carbamate ester functional-class name.

    Scope is one neutral acyclic C(=O)(N)(O) group with one proved organyl arm
    on O and zero to two on N. Every other heavy atom must belong to one of
    these mapped branches. The method declines core labels
    except a carbon isotope on its sole carbonyl carbon, isotope-labelled noncarbon atoms, explicit isotope
    hydrogen, charges, radicals, or enhanced stereochemistry.
    """
    core = _carbamate_core(mol, Chem)
    if core is not None and mol.GetAtomWithIdx(core[3]).IsInRing():
        from .annular_carbamate_carboxylate_methods import (
            annular_carbamate_carboxylate_decision,
        )
        return annular_carbamate_carboxylate_decision(mol, Chem, core)
    if mol.GetStereoGroups():
        return None
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           for atom in iter_atoms(mol)):
        return None
    if any(atom.GetAtomicNum() == 1 and atom.GetIsotope()
           for atom in iter_atoms(mol)):
        return None
    if any(atom.GetIsotope() and atom.GetAtomicNum() != 6
           for atom in iter_atoms(mol)):
        return None

    if core is None:
        return None
    carbonyl, carbonyl_o, ester_o, nitrogen = core
    core_atoms = frozenset(core)
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in core_atoms):
        return None
    co = mol.GetAtomWithIdx(carbonyl_o)
    bridge = mol.GetAtomWithIdx(ester_o)
    n_atom = mol.GetAtomWithIdx(nitrogen)
    if (co.GetFormalCharge() or co.GetDegree() != 1 or co.GetIsotope()
            or int(co.GetChiralTag())):
        return None
    if (bridge.GetFormalCharge() or bridge.GetDegree() != 2
            or bridge.GetIsotope() or int(bridge.GetChiralTag())
            or bridge.IsInRing()):
        return None
    if (n_atom.GetFormalCharge() or n_atom.IsInRing()
            or int(n_atom.GetChiralTag())):
        return None
    if (mol.GetBondBetweenAtoms(carbonyl, carbonyl_o).GetBondTypeAsDouble() != 2
            or any(mol.GetBondBetweenAtoms(carbonyl, i).GetBondTypeAsDouble() != 1
                   for i in (ester_o, nitrogen))):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           for i in core_atoms if i != carbonyl):
        return None

    oxygen_roots = [atom.GetIdx() for atom in bridge.GetNeighbors()
                    if atom.GetIdx() != carbonyl]
    if len(oxygen_roots) != 1 or mol.GetAtomWithIdx(oxygen_roots[0]).GetAtomicNum() != 6:
        return None
    oxygen_name, oxygen_atoms = _proved_organyl(
        mol, Chem, oxygen_roots[0], ester_o)
    haloalkyl_extension = False
    if not oxygen_name and allow_haloalkyl:
        oxygen_name, oxygen_atoms = _haloalkyl_organyl(
            mol, Chem, oxygen_roots[0], ester_o)
        haloalkyl_extension = bool(oxygen_name)
    if not oxygen_name:
        return None
    if haloalkyl_extension and any(atom.GetIsotope() for atom in iter_atoms(mol)):
        # A core or N-arm isotope is a separate multi-component comparison;
        # it must not inherit this initially unlabeled source proof.
        return None

    n_roots = [atom.GetIdx() for atom in n_atom.GetNeighbors()
               if atom.GetIdx() != carbonyl]
    if len(n_roots) > 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8}
                               for i in n_roots):
        return None
    n_names, n_owned = [], set()
    for root in n_roots:
        name, owned = _proved_organyl(mol, Chem, root, nitrogen, allow_n_oxygen=True)
        if not name or n_owned & owned:
            return None
        n_names.append(name)
        n_owned.update(owned)
    if haloalkyl_extension:
        # This new source-bounded family initially combines only a free NH2
        # or one simple N-alkyl arm with the proved O-component. More complex
        # N arms continue through their pre-existing routes.
        if (len(n_roots) > 1 or n_roots and
                (len(n_owned) > 3 or not _unbranched_saturated_carbon_path(
                    mol, n_owned, n_roots[0]))):
            return None

    expected = set(core_atoms) | set(oxygen_atoms) | n_owned
    heavy = {atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1}
    if expected != heavy or (set(oxygen_atoms) & n_owned):
        return None

    # Class seniority is checked independently of the local motif recognizer.
    # Lower-ranked groups may belong to one of the independently proved
    # organyl branches. A group spanning components cannot be silently lost.
    inventory = chemical_inventory(mol)
    if not inventory.supported and not any(atom.GetIsotope() for atom in iter_atoms(mol)):
        # The extended inventory can classify neutral heteroaromatic S/P
        # atoms that the original narrow inventory declines. Do not erase
        # unsupported isotope roles merely to enter this broader scope.
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != "carbamate"
            or inventory.principal[0].center != carbonyl):
        return None
    principal = inventory.principal[0]
    if any(group.center != carbonyl and (
            group.seniority <= principal.seniority
            or not (set(group.atoms) <= set(oxygen_atoms)
                    or set(group.atoms) <= n_owned)) for group in inventory.groups):
        return None

    carbamate_name = render_acyclic_carbamate(
        mol, Chem, nitrogen, frozenset({carbonyl, carbonyl_o, nitrogen}), n_names)
    if not carbamate_name:
        return None
    name = oxygen_name + " " + carbamate_name
    centers = frozenset({carbonyl})
    atoms = frozenset(heavy)
    # Mapped branches may own unspecified stereogenic units. Preserve that
    # source uncertainty in the public selection result rather than implying
    # that a particular configuration was chosen by this producer.
    from .naming_result import SelectionLimitation
    stereo_issues = ((SelectionLimitation(
        "stereochemistry", "P-91.2.1.2.1/P-91.2.2",
        "One or more potential stereogenic units have no source configuration."),)
        if any(item.specified != Chem.StereoSpecified.Specified
               for item in Chem.FindPotentialStereo(mol)) else ())
    if haloalkyl_extension:
        stereo_issues += (SelectionLimitation(
            "parent-and-method", "P-44/P-45/P-51",
            "The O-organyl and carbamate construction is mapped; complete "
            "whole-name parent and method comparisons remain unproved."),)
    return MethodDecision(
        name=name,
        method="functional-class",
        principal_centers=centers,
        atoms=atoms,
        rule=_RULE,
        component_roles=(
            ("carbamate-core", core_atoms - {ester_o}, ()),
            ("O-organyl", oxygen_atoms | {ester_o}, ()),
            ("N-organyl", frozenset(n_owned), ()),
        ),
        preference_issues=stereo_issues,
    )


def render_acyclic_carbamate(mol, Chem, nitrogen, parent_atoms, n_names):
    # P-14.3.4.4/P-65.2.1.1: a single substituent has only one N site.
    # Separate locants distinguish two N arms from one compound arm.
    single = len(n_names) == 1
    construction = MappedComponent(
        parent="carbamate",
        atom_locants=tuple((idx, "N" if idx == nitrogen else "")
                           for idx in sorted(parent_atoms)),
        atoms=parent_atoms,
        prefixes=tuple(("" if single else "N", prefix) for prefix in n_names),
        prefix_mode="unlocanted" if single else "locanted",
        nitrogen_site_locants=not single,
    )
    return decorate_component(mol, Chem, construction)
