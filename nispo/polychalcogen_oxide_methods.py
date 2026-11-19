"""Mapped substitutive names for bounded P-68.4.3.2 chalcogen oxides."""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT
from .formatting import format_prefixes, multiplied_suffix, prefix_alphabetization_key
from .naming_methods import MethodDecision


RULE = "P-68.4.3.2/P-14.1.3/P-14.3.4.4/P-14.4(c,f,g,h)/P-29.3/P-51.1.1/P-51.4.1.1/P-82.2.2.1/P-92.1.2.1.1"
_ELEMENTS = {16: "sulfane", 34: "selane", 52: "tellane"}


@dataclass(frozen=True, slots=True)
class PolyChalcogenOxideSite:
    elements: tuple[int, ...]
    chain: tuple[int, ...]
    oxo_by_position: tuple[tuple[int, ...], ...]
    terminal_roots: tuple[int, int]
    terminal_atoms: tuple[frozenset[int], frozenset[int]]


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _ordered_path(mol, atoms):
    adjacency = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() in atoms) for i in atoms}
    if any(len(row) > 2 for row in adjacency.values()):
        return ()
    ends = [i for i, row in adjacency.items() if len(row) == 1]
    if len(ends) != 2:
        return ()
    path, previous, current = [], -1, min(ends)
    while True:
        path.append(current)
        options = [i for i in adjacency[current] if i != previous]
        if not options:
            break
        if len(options) != 1:
            return ()
        previous, current = current, options[0]
        if current in path:
            return ()
    return tuple(path) if len(path) == len(atoms) else ()


def _carbon_organyl(mol, root, attached):
    """Return a saturated carbon tree or single phenyl branch, otherwise None."""
    if not _single(mol, root, attached):
        return None
    atoms, pending = {root}, [root]
    while pending:
        i = pending.pop()
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j == attached or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            if j not in atoms:
                atoms.add(j)
                pending.append(j)
    outside = {(i, n.GetIdx()) for i in atoms
               for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
    if outside != {(root, attached)}:
        return None
    rings = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if rings:
        if (len(rings) != 6 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in rings)
                or len(atoms - rings) not in {0, 1, 2}):
            return None
        internal = [b for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        ring_bonds = [b for b in internal if b.GetBeginAtomIdx() in rings
                      and b.GetEndAtomIdx() in rings]
        outside_bonds = [b for b in internal if b.GetBeginAtomIdx() not in rings
                         or b.GetEndAtomIdx() not in rings]
        if (len(ring_bonds) != 6 or not all(b.GetIsAromatic() for b in ring_bonds)
                or len(outside_bonds) != len(atoms - rings)
                or any(b.GetBondTypeAsDouble() != 1 for b in outside_bonds)):
            return None
        if len(atoms - rings) and sum(1 for b in outside_bonds
                if (b.GetBeginAtomIdx() in rings) != (b.GetEndAtomIdx() in rings)) != 1:
            return None
        if len(atoms - rings) == 2:
            nonring_edges = [b for b in outside_bonds
                             if b.GetBeginAtomIdx() not in rings and b.GetEndAtomIdx() not in rings]
            if len(nonring_edges) != 1:
                return None
    else:
        internal = [b for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
        if len(internal) != len(atoms) - 1 or not all(b.GetBondTypeAsDouble() == 1 for b in internal):
            return None
    return frozenset(atoms)


def _recognize(mol):
    if (mol.GetNumAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol))
            or any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol))):
        return None
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    es = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() in _ELEMENTS}
    oxygens = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 8}
    carbons = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    if not es or not carbons or es | oxygens | carbons != heavy or len(es) not in MULT:
        return None
    if not 2 <= len(es) <= 6:
        return None
    chain = _ordered_path(mol, es)
    if not chain or any(not _single(mol, a, b) for a, b in zip(chain, chain[1:])):
        return None
    element_set = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain}
    if len(element_set) != 1:
        return None
    x_by_position, oxo_by_position = [], []
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.IsInRing() or atom.GetIsotope() or int(atom.GetChiralTag())
                or atom.GetIsAromatic()):
            return None
        oxo = []
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 8:
                bond = mol.GetBondBetweenAtoms(idx, j)
                if (neighbor.GetIsotope() or neighbor.GetFormalCharge() or neighbor.IsInRing()
                        or neighbor.GetDegree() != 1 or bond.GetBondTypeAsDouble() != 2.0):
                    return None
                oxo.append(j)
            elif neighbor.GetAtomicNum() not in {6, 16, 34, 52}:
                if neighbor.GetAtomicNum() != 1:
                    return None
        if len(oxo) not in {1, 2}:
            return None
        x_by_position.append(len(oxo))
        oxo_by_position.append(tuple(sorted(oxo)))
    roots, terminal_atoms = [], []
    for idx in (chain[0], chain[-1]):
        carbon_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if n.GetAtomicNum() == 6]
        if len(carbon_neighbors) != 1:
            return None
        root = carbon_neighbors[0]
        branch = _carbon_organyl(mol, root, idx)
        if branch is None:
            return None
        roots.append(root)
        terminal_atoms.append(branch)
    if terminal_atoms[0] & terminal_atoms[1]:
        return None
    if terminal_atoms[0] | terminal_atoms[1] | es | oxygens != heavy:
        return None
    for atom in iter_atoms(mol):
        z, idx = atom.GetAtomicNum(), atom.GetIdx()
        if atom.GetIsotope() and (z != 6 or idx not in terminal_atoms[0] | terminal_atoms[1]
                                   or atom.GetIsotope() != 13):
            return None
        if int(atom.GetChiralTag()) and (z != 6 or idx not in terminal_atoms[0] | terminal_atoms[1]):
            return None
        if z == 6 and atom.GetIsAromatic() and idx not in terminal_atoms[0] | terminal_atoms[1]:
            return None
        if int(atom.GetChiralTag()) and atom.GetIsotope() and atom.GetIsotope() != 13:
            return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    # A neutral chalcogen center with one E-chain bond, one terminal C bond at
    # each end, and x oxo double bonds has lambda bonding number 2x + 2.
    for pos, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        bonding_number = sum(mol.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondTypeAsDouble()
                             for neighbor in atom.GetNeighbors())
        if atom.GetTotalNumHs(includeNeighbors=True) or bonding_number != 2 * x_by_position[pos] + 2:
            return None
    return PolyChalcogenOxideSite(
        tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain), chain,
        tuple(oxo_by_position), tuple(roots), tuple(terminal_atoms))


def recognize(mol):
    """Return a structurally proved site without rendering its name."""
    return _recognize(mol)


def _branch_signature(mol, Chem, root, atoms, *, strip_isotopes=False, strip_stereo=False):
    copy = Chem.Mol(mol)
    if strip_isotopes:
        for atom in copy.GetAtoms():
            atom.SetIsotope(0)
    if strip_stereo:
        for atom in copy.GetAtoms():
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in copy.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolFragmentToSmiles(copy, atomsToUse=sorted(atoms), rootedAtAtom=root,
                                    canonical=True, isomericSmiles=True)


def _branch_rows(mol, Chem, site):
    from .component_mapping import mapped_branch

    rows = []
    for root, atoms, blocked in zip(site.terminal_roots, site.terminal_atoms,
                                    (site.chain[0], site.chain[-1]), strict=True):
        text, owned = mapped_branch(mol, Chem, root, blocked)
        if not text or frozenset(i for i in owned if mol.GetAtomWithIdx(i).GetAtomicNum() > 1) != atoms:
            return None
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        rows.append((text, atoms, root))
    if len(rows) != 2:
        return None
    same_constitution = (
        _branch_signature(mol, Chem, rows[0][2], rows[0][1], strip_isotopes=True,
                          strip_stereo=True)
        == _branch_signature(mol, Chem, rows[1][2], rows[1][1], strip_isotopes=True,
                             strip_stereo=True))
    if same_constitution:
        # P-82.2.2.1 resolves modified versus natural equivalent branches.
        # Stereo-only and differently isotope-modified pairs remain outside
        # this operation because their relative ordering is not established.
        has_stereo = [any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in row[1])
                      for row in rows]
        if any(has_stereo):
            return None
        stereo_signatures = [_branch_signature(mol, Chem, row[2], row[1],
                                                strip_isotopes=True) for row in rows]
        isotope_signatures = [_branch_signature(mol, Chem, row[2], row[1],
                                                strip_stereo=True) for row in rows]
        if stereo_signatures[0] != stereo_signatures[1]:
            return None
        if isotope_signatures[0] != isotope_signatures[1]:
            modified = [any(mol.GetAtomWithIdx(i).GetIsotope() for i in row[1])
                        for row in rows]
            if modified[0] == modified[1]:
                return None
    return tuple(rows)


def _prefix_key(row, mol):
    text, atoms, _ = row
    key = prefix_alphabetization_key(text)
    modified = any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms)
    return (*key[:3], not modified, key[3])


def _candidate_key(site, rows, reverse, mol):
    n = len(site.chain)
    order = tuple(reversed(range(n))) if reverse else tuple(range(n))
    x = [len(site.oxo_by_position[i]) for i in order]
    suffix_locs = tuple(sorted(pos for pos, amount in enumerate(x, 1)
                               for _ in range(amount)))
    branch_at = {1: rows[1] if reverse else rows[0],
                 n: rows[0] if reverse else rows[1]}
    alphabetized = sorted(branch_at.items(), key=lambda item: _prefix_key(item[1], mol))
    # P-14.5 compares the complete prefix names; P-82.2.2.1 overrides that
    # comparison for modified versus unmodified otherwise-identical branches.
    prefix_assignment = tuple(loc for loc, _ in alphabetized)
    lambda_priority = tuple(-((2 * x[i] + 2)) for i in range(n))
    return suffix_locs, prefix_assignment, lambda_priority


def polychalcogen_oxide_decision(mol, Chem=None):
    """Return a complete mapped substitutive decision for a bounded graph."""
    if Chem is None:
        from rdkit import Chem
    site = _recognize(mol)
    if site is None:
        return None
    rows = _branch_rows(mol, Chem, site)
    if rows is None:
        return None
    n = len(site.chain)
    # Build both end numberings using source criteria: suffix locants first,
    # alpha-prefix order for a remaining tie, then P-14.4(h) lambda order.
    orientations = [(_candidate_key(site, rows, reverse, mol), reverse)
                    for reverse in (False, True)]
    _, reverse = min(orientations, key=lambda item: item[0])
    position_order = tuple(reversed(range(n))) if reverse else tuple(range(n))
    ordered_chain = tuple(site.chain[i] for i in position_order)
    x = tuple(len(site.oxo_by_position[i]) for i in position_order)
    position_for_atom = {idx: str(pos) for pos, idx in enumerate(ordered_chain, 1)}
    oxo_locants = tuple(sorted(pos for pos, amount in enumerate(x, 1)
                               for _ in range(amount)))
    suffix = multiplied_suffix(len(oxo_locants), "one")
    if not suffix or len(oxo_locants) == 0:
        return None
    lambda_text = ",".join(f"{pos}lambda{2 * x[pos - 1] + 2}"
                           for pos in range(1, n + 1))
    element = site.elements[0]
    parent = f"{lambda_text}-{MULT.get(n, '')}{_ELEMENTS[element]}"
    # Locate the endpoint arm corresponding to each selected chain end.
    ends = {"1": rows[1] if reverse else rows[0],
            str(n): rows[0] if reverse else rows[1]}
    prefix_items = [(loc, row[0]) for loc, row in ends.items()]
    equivalent = (_branch_signature(mol, Chem, ends["1"][2], ends["1"][1],
                                    strip_isotopes=True, strip_stereo=True)
                  == _branch_signature(mol, Chem, ends[str(n)][2], ends[str(n)][1],
                                       strip_isotopes=True, strip_stereo=True))
    isotope_differ = (equivalent and
        any(mol.GetAtomWithIdx(i).GetIsotope() for i in ends["1"][1]) !=
        any(mol.GetAtomWithIdx(i).GetIsotope() for i in ends[str(n)][1]))
    x_mixed_dinuclear = n == 2 and x[0] != x[1]
    if (x_mixed_dinuclear and ends["1"][0] == ends[str(n)][0]
            and not any(mark in ends["1"][0] for mark in "()[]{}")):
        # P-68.4.3.2 prints the mixed dioxide/trioxide family with the
        # identical terminal prefix unlocanted (P-14.3.4.4).
        prefix = f"di{ends['1'][0]}"
    elif isotope_differ:
        # P-82.2.2.1 requires the modified otherwise-identical substituent to
        # precede its natural counterpart, including when locants can reverse.
        ordered = sorted(prefix_items, key=lambda item: _prefix_key(ends[item[0]], mol))
        prefix = "-".join(_render_prefix_item(item[0], item[1]) for item in ordered)
    else:
        prefix = format_prefixes(prefix_items, hierarchical_enclosures=True)
    name = f"{prefix}-{parent}-{','.join(map(str, oxo_locants))}-{suffix}"
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    core_atoms = frozenset(site.chain) | frozenset(i for group in site.oxo_by_position for i in group)
    core_map = []
    for idx in ordered_chain:
        loc = position_for_atom[idx]
        core_map.append((idx, loc))
        source_position = site.chain.index(idx)
        core_map.extend((oxygen, loc) for oxygen in site.oxo_by_position[source_position])
    roles = [("polychalcogen-oxide-parent", core_atoms, tuple(core_map))]
    roles.extend((f"terminal-organyl:{i}", row[1], ()) for i, row in enumerate(
        [ends["1"], ends[str(n)]], 1))
    flat = [idx for _, atoms, _ in roles for idx in atoms]
    if len(flat) != len(set(flat)) or frozenset(flat) != all_heavy:
        return None
    return MethodDecision(name=name, method="substitutive",
        principal_centers=frozenset(site.chain), atoms=all_heavy, rule=RULE,
        atom_locants=tuple(core_map), component_roles=tuple(roles))


standardMethodDecision = polychalcogen_oxide_decision


def _render_prefix_item(locant, text):
    """Render a component prefix with P-16.5 hierarchical enclosure."""
    return format_prefixes([(locant, text)], hierarchical_enclosures=True)
