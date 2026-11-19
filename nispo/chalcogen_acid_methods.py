"""Bounded mapped suffix names for neutral Se/Te acids and amides.

The producer covers one P-65.3.1 oxyacid or P-66.1.1.2 oxyamide attached to
one atom of a saturated, unbranched, acyclic carbon parent. Original atom
indices are retained through component decoration and the returned method
decision owns every heavy atom exactly once.
"""

from dataclasses import dataclass, replace

from .molecule_iteration import iter_atoms, iter_bonds
from .branch_rules import parent_chain_name
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .formatting import attach_suffix, complete_parent_name
from .naming_methods import MethodDecision
from .parent_selection import graph_parent_facts, compare_parents, parent_frontier


_ACID_SUFFIX = {
    # Nested keys keep class 7a Se/Te acids after sulfinic acid (7,2) and
    # before the current class-7b carbonic acid key (7,3).
    (34, 2): ("selenonic acid", (7, 2, 1), "P-65.3.1/P-41/P-42.1"),
    (34, 1): ("seleninic acid", (7, 2, 2), "P-65.3.1/P-41/P-42.1"),
    (52, 2): ("telluronic acid", (7, 2, 3), "P-65.3.1/P-41/P-42.1"),
    (52, 1): ("tellurinic acid", (7, 2, 4), "P-65.3.1/P-41/P-42.1"),
}

# P-41 class 11 orders amides by their corresponding acid. The extra tuple
# component preserves the sulfur helper's existing relative position in this
# internal comparable key while following P-42.1 for Se and Te.
_AMIDE_SUFFIX = {
    (34, 2): ("selenonamide", (11, 1, 2), "P-66.1.1.2/P-41"),
    (34, 1): ("seleninamide", (11, 1, 3), "P-66.1.1.2/P-41"),
    (52, 2): ("telluronamide", (11, 1, 4), "P-66.1.1.2/P-41"),
    (52, 1): ("tellurinamide", (11, 1, 5), "P-66.1.1.2/P-41"),
}


@dataclass(frozen=True, slots=True)
class _Unit:
    kind: str
    center: int
    host: int
    oxo: tuple[int, ...]
    ligand: int
    n_substituents: tuple[int, ...]
    suffix: str
    rank: tuple[int, ...]
    rule: str


def _bond(mol, left, right, order):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == order


def _terminal_oxo(mol, center):
    return tuple(sorted(a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()
                        if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                        and not a.GetFormalCharge() and not a.GetIsotope()
                        and _bond(mol, center, a.GetIdx(), 2)))


def _recognize_unit(mol):
    """Return one exact neutral Se/Te acid/amide unit, else ``None``."""
    centers = [a for a in iter_atoms(mol) if a.GetAtomicNum() in {34, 52}]
    if len(centers) != 1:
        return None
    center = centers[0]
    eidx, element = center.GetIdx(), center.GetAtomicNum()
    if (center.GetFormalCharge() or center.GetNumRadicalElectrons()
            or center.GetIsotope() or int(center.GetChiralTag()) or center.IsInRing()):
        return None
    oxo = _terminal_oxo(mol, eidx)
    if len(oxo) not in {1, 2}:
        return None
    neighbors = tuple(center.GetNeighbors())
    carbon = [a.GetIdx() for a in neighbors if a.GetAtomicNum() == 6
              and _bond(mol, eidx, a.GetIdx(), 1)]
    single_o = [a.GetIdx() for a in neighbors if a.GetAtomicNum() == 8
                and _bond(mol, eidx, a.GetIdx(), 1)]
    nitrogen = [a.GetIdx() for a in neighbors if a.GetAtomicNum() == 7
                and _bond(mol, eidx, a.GetIdx(), 1)]
    if len(carbon) != 1 or len(oxo) + len(carbon) + len(single_o) + len(nitrogen) != len(neighbors):
        return None
    if len(neighbors) != len(oxo) + 2:
        return None
    if len(single_o) == 1 and not nitrogen:
        oxygen = mol.GetAtomWithIdx(single_o[0])
        if (oxygen.GetDegree() != 1 or oxygen.GetFormalCharge()
                or oxygen.GetIsotope() or int(oxygen.GetChiralTag())
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        item = _ACID_SUFFIX.get((element, len(oxo)))
        if item is None:
            return None
        suffix, rank, rule = item
        return _Unit("acid", eidx, carbon[0], oxo, single_o[0], (), suffix, rank, rule)
    if len(nitrogen) == 1 and not single_o:
        nidx = nitrogen[0]
        n_atom = mol.GetAtomWithIdx(nidx)
        if (n_atom.GetFormalCharge() or n_atom.GetNumRadicalElectrons()
                or n_atom.GetIsotope() or int(n_atom.GetChiralTag()) or n_atom.IsInRing()):
            return None
        if not _bond(mol, eidx, nidx, 1):
            return None
        carbon_subs = []
        for atom in n_atom.GetNeighbors():
            if atom.GetIdx() == eidx:
                continue
            if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                    or not _bond(mol, nidx, atom.GetIdx(), 1)):
                return None
            carbon_subs.append(atom.GetIdx())
        if len(carbon_subs) > 2 or n_atom.GetDegree() > 3:
            return None
        item = _AMIDE_SUFFIX.get((element, len(oxo)))
        if item is None:
            return None
        suffix, rank, rule = item
        return _Unit("amide", eidx, carbon[0], oxo, nidx,
                     tuple(sorted(carbon_subs)), suffix, rank,
                     rule + "/P-66.1.1.3.1.1")
    return None


def _carbon_parent_path(mol, host):
    """Find the complete unbranched saturated C component containing host."""
    parent = {host}
    stack = [host]
    while stack:
        idx = stack.pop()
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if neighbor.GetAtomicNum() != 6:
                continue
            if not _bond(mol, idx, other, 1):
                return None
            if other not in parent:
                parent.add(other)
                stack.append(other)
    if (len(parent) > 12 or any(mol.GetAtomWithIdx(i).IsInRing() for i in parent)
            or any(sum(n.GetIdx() in parent for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2
                   for i in parent)):
        return None
    ends = [i for i in parent if sum(n.GetIdx() in parent
                                    for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1]
    if len(parent) == 1:
        return (host,)
    if len(ends) != 2:
        return None
    paths = []
    for start in ends:
        path, previous, current = [], -1, start
        while True:
            path.append(current)
            choices = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                       if n.GetIdx() in parent and n.GetIdx() != previous]
            if not choices:
                break
            if len(choices) != 1:
                return None
            previous, current = current, choices[0]
        if set(path) == parent and host in path:
            paths.append(tuple(path))
    return paths[0] if paths else None


def _branch_atoms_are_bounded(mol, atom_ids):
    return (bool(atom_ids) and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                                   and not mol.GetAtomWithIdx(i).IsInRing()
                                   for i in atom_ids)
            and all(b.GetBondTypeAsDouble() == 1 for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in atom_ids and b.GetEndAtomIdx() in atom_ids))


def _unsupported_annotation(mol, unit, parent_atoms, branch_atoms):
    owned = set(parent_atoms) | set(branch_atoms)
    owned |= {unit.center, unit.ligand, *unit.oxo}
    for atom in iter_atoms(mol):
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        if number == 1:
            if atom.GetIsotope():
                return True
            continue
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return True
        if idx not in owned:
            return True
        if atom.GetIsotope() and (number != 6 or idx in {unit.center, unit.ligand, *unit.oxo}):
            return True
        if int(atom.GetChiralTag()) and idx not in set(parent_atoms) | set(branch_atoms):
            return True
    return bool(mol.GetStereoGroups())


def chalcogen_acid_decision(mol, Chem):
    """Return a complete mapped Se/Te acid/amide suffix decision or ``None``.

    Supported ordinary annotations are limited to carbon isotope/stereo in
    the carbon parent and mapped N-alkyl groups. The route requires exactly
    one Se/Te center and declines additional functions, charges, unsupported
    central-atom annotations, cycles, branches from the parent carbon chain,
    and any atom not accounted for by the typed components.
    """
    if (not mol.GetNumAtoms() or mol.GetNumHeavyAtoms() > 64
            or mol.GetStereoGroups()):
        return None
    unit = _recognize_unit(mol)
    if unit is None:
        return None
    parent_path = _carbon_parent_path(mol, unit.host)
    if not parent_path:
        return None
    parent_set = frozenset(parent_path)
    if unit.host not in parent_set:
        return None
    branches = []
    branch_owned = set()
    if unit.kind == "amide":
        for root in unit.n_substituents:
            name, atoms = mapped_branch(mol, Chem, root, unit.ligand)
            atoms = frozenset(atoms)
            if (not name or root not in atoms or not _branch_atoms_are_bounded(mol, atoms)
                    or branch_owned & set(atoms) or set(atoms) & parent_set):
                return None
            branches.append((name, atoms, root))
            branch_owned.update(atoms)
    elif unit.ligand != -1:
        pass
    if _unsupported_annotation(mol, unit, parent_set, branch_owned):
        return None
    # The atom-level proof above is strengthened by requiring that every
    # non-parent carbon belongs to exactly one amide-N branch.
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    core_atoms = {unit.center, unit.ligand, *unit.oxo}
    if parent_set | branch_owned | core_atoms != all_heavy:
        return None
    if (len(parent_set & branch_owned) or parent_set & core_atoms
            or branch_owned & core_atoms):
        return None

    candidates = []
    orientations = (parent_path, tuple(reversed(parent_path)))
    for path in orientations:
        locants = {idx: str(i) for i, idx in enumerate(path, 1)}
        site = locants[unit.host]
        parent_carbon_labels = [mol.GetAtomWithIdx(i).GetIsotope() for i in path]
        uniformly_labelled_parent = (
            bool(parent_carbon_labels) and all(parent_carbon_labels)
            and len(set(parent_carbon_labels)) == 1
        )
        partial_parent_isotope_locant = (
            len(path) > 1 and any(parent_carbon_labels)
            and not uniformly_labelled_parent
        )
        # P-82.6.1.1 requires all normally omitted locants when a partial
        # isotopic descriptor needs locants. P-82.6.1.3 permits omission when
        # every atom of this parent element is modified in the same way.
        omit = (len(path) == 1 or (len(path) == 2 and site == "1"
                                  and not partial_parent_isotope_locant))
        parent = complete_parent_name(parent_chain_name(mol, path))
        name_parent = attach_suffix(parent, unit.suffix,
                                    () if omit else (site,))
        prefix_pairs = tuple(("N", branch_name) for branch_name, _, _ in branches)
        atom_map = tuple((idx, locants[idx]) for idx in path)
        atom_map += ((unit.center, ""),) + tuple((idx, "") for idx in unit.oxo)
        atom_map += ((unit.ligand, "O" if unit.kind == "acid" else "N"),)
        component_atoms = frozenset(parent_set | core_atoms)
        embedded = frozenset(branch_owned)
        component = MappedComponent(
            parent=name_parent,
            atom_locants=atom_map,
            atoms=component_atoms,
            prefixes=prefix_pairs,
            embedded_atoms=embedded,
            systematic_parent=True,
            nitrogen_site_locants=unit.kind == "amide",
        )
        name = decorate_component(mol, Chem, component)
        if not name:
            continue
        facts = graph_parent_facts(
            mol, path, locants, principal_class=unit.rank,
            suffix_centers=(unit.center,), suffix_locants=(site,),
            prefixes=prefix_pairs, kind="chain", name=name,
            allow_annotations=True,
        )
        if facts is None:
            continue
        # The suffix core is explicitly owned above; the only remainder is
        # ordinary carbon in independently rendered N-alkyl branches.
        facts = replace(facts, substituent_nonstandard_key=())
        roles = [("parent-hydride", parent_set,
                  tuple((idx, locants[idx]) for idx in path))]
        local = frozenset(core_atoms)
        core_locs = ((unit.center, ""),) + tuple((idx, "") for idx in unit.oxo)
        core_locs += ((unit.ligand, "O" if unit.kind == "acid" else "N"),)
        roles.append((unit.kind, local, core_locs))
        for _, atoms, _ in branches:
            roles.append(("amide-n-substituent", atoms,
                          tuple((idx, "") for idx in sorted(atoms))))
        if (len({i for _, atoms, _ in roles for i in atoms}) != len(all_heavy)
                or frozenset(i for _, atoms, _ in roles for i in atoms) != all_heavy):
            continue
        candidates.append((name, facts, component, tuple(roles), site))
    frontier = parent_frontier(candidates, lambda row: row[1])
    if not frontier:
        return None
    if len({row[0] for row in frontier}) != 1:
        return None
    for i, left in enumerate(frontier):
        for right in frontier[i + 1:]:
            if compare_parents(left[1], right[1]).preference not in {-1, 0}:
                return None
    name, facts, component, roles, site = frontier[0]
    role_atoms = [idx for _, atoms, _ in roles for idx in atoms]
    if len(role_atoms) != len(set(role_atoms)) or frozenset(role_atoms) != all_heavy:
        return None
    return MethodDecision(
        name=name,
        method="substitutive",
        principal_centers=frozenset({unit.center}),
        atoms=all_heavy,
        rule=unit.rule + "/P-44/P-45",
        atom_locants=component.atom_locants,
        component_roles=roles,
    )
