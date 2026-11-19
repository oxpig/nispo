"""Mapped, bounded P-68.4.1.3 homogeneous chalcogen-chain pseudoketones.

The helper never treats the homogeneous chalcogen path as an ester linker.
It admits exact neutral path graphs with simple acyl and organyl groups,
and returns a fully owned MethodDecision for source-proved constructions.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS
from .naming_methods import MethodDecision


RULE = "P-41/P-44/P-45/P-51.3/P-64.1.2.1/P-64.3/P-65.7.5.1/P-68.4.1.3"
_CHALCOGEN = {
    8: ("oxygen", "oxidane", "oxidanyl"),
    16: ("sulfur", "sulfane", "sulfanyl"),
    34: ("selenium", "selane", "selanyl"),
    52: ("tellurium", "tellane", "tellanyl"),
}
_PREFIX_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
                7: "hepta", 8: "octa", 9: "nona", 10: "deca"}


@dataclass(frozen=True, slots=True)
class PseudoketoneSite:
    element: int
    chain: tuple[int, ...]
    carbonyls: tuple[int, ...]
    carbonyl_oxygen: tuple[int, ...]
    carbon_components: tuple[frozenset[int], ...]
    alkyl_counts: tuple[int, ...]
    terminal_organyl_atoms: frozenset[int]
    terminal_organyl_name: str | None


def _single(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _path(mol, atoms):
    adjacency = {i: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                           if n.GetIdx() in atoms) for i in atoms}
    if any(len(v) > 2 for v in adjacency.values()):
        return ()
    ends = sorted(i for i, v in adjacency.items() if len(v) == 1)
    if len(ends) != 2:
        return ()
    result, previous, current = [], None, ends[0]
    while True:
        if current in result:
            return ()
        result.append(current)
        nxt = [i for i in adjacency[current] if i != previous]
        if not nxt:
            break
        if len(nxt) != 1:
            return ()
        previous, current = current, nxt[0]
    if len(result) != len(atoms) or result[-1] != ends[1]:
        return ()
    return tuple(result)


def _simple_acyl(mol, carbonyl, chalcogen, carbon_set):
    """Return (owned C atoms, count) for a linear saturated R-C(=O)-X unit."""
    carbon = mol.GetAtomWithIdx(carbonyl)
    oxygens = [n.GetIdx() for n in carbon.GetNeighbors()
               if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(carbonyl, n.GetIdx()).GetBondTypeAsDouble() == 2]
    neighbors = [n.GetIdx() for n in carbon.GetNeighbors()
                 if n.GetAtomicNum() > 1 and n.GetIdx() != chalcogen
                 and n.GetIdx() not in oxygens]
    if len(oxygens) != 1 or len(neighbors) != 1 or not _single(mol, carbonyl, chalcogen):
        return None
    root = neighbors[0]
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6 or root not in carbon_set:
        return None
    seen, pending = {root}, [root]
    while pending:
        idx = pending.pop()
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsAromatic() or atom.IsInRing():
            return None
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni == carbonyl:
                if not _single(mol, idx, ni):
                    return None
                continue
            if nbr.GetAtomicNum() != 6:
                return None
            if ni not in seen:
                if not _single(mol, idx, ni):
                    return None
                seen.add(ni)
                pending.append(ni)
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() or mol.GetAtomWithIdx(i).IsInRing()
           for i in seen):
        return None
    if sum(1 for b in iter_bonds(mol) if b.GetBeginAtomIdx() in seen
           and b.GetEndAtomIdx() in seen) != len(seen) - 1:
        return None
    # The source-backed initial component grammar is an unbranched alkyl chain.
    if any(sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetAtomicNum() == 6 and n.GetIdx() in seen) > 2 for i in seen):
        return None
    endpoints = [i for i in seen if sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                                        if n.GetAtomicNum() == 6 and n.GetIdx() in seen) <= 1]
    expected_ends = 1 if len(seen) == 1 else 2
    root_degree = sum(1 for n in mol.GetAtomWithIdx(root).GetNeighbors()
                      if n.GetAtomicNum() == 6 and n.GetIdx() in seen)
    if len(endpoints) != expected_ends or root_degree > 1 or any(
            mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in seen):
        return None
    return frozenset(seen | {carbonyl, oxygens[0]}), len(seen) + 1, oxygens[0]


def recognize_chalcogen_chain_pseudoketone(mol):
    """Recognize one neutral homogeneous chalcogen path with acyl ends."""
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           or int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    # Carbonyl oxygens are not members of the homogeneous chain even though
    # oxygen is itself a chalcogen. Classify the chain by its single-bonded
    # path; a doubly bonded carbonyl O is excluded from that candidate set.
    candidates = []
    for element in _CHALCOGEN:
        chain_atoms = {a.GetIdx() for a in iter_atoms(mol)
                       if a.GetAtomicNum() == element
                       and not any(b.GetBondTypeAsDouble() > 1
                                   for b in a.GetBonds())}
        chain = _path(mol, chain_atoms) if len(chain_atoms) >= 3 else ()
        if chain:
            candidates.append((element, chain_atoms, chain))
    if len(candidates) != 1:
        return None
    element, chain_atoms, chain = candidates[0]
    if len(chain) < 3 or any(not _single(mol, a, b) for a, b in zip(chain, chain[1:])):
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in chain):
        return None
    if any(a.GetAtomicNum() not in {1, 6, element, 8} for a in iter_atoms(mol)):
        return None
    terminal = (chain[0], chain[-1])
    carbonyls, oxygens, components, counts = [], [], [], []
    terminal_organyl_atoms = frozenset()
    terminal_organyl_name = None
    carbon_atoms = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    for position, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        xneighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == element]
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if idx not in terminal:
            if len(xneighbors) != 2 or len(heavy_neighbors) != 2:
                return None
            continue
        if len(xneighbors) != 1 or len(heavy_neighbors) > 2:
            return None
        other = [n.GetIdx() for n in heavy_neighbors if n.GetIdx() not in xneighbors]
        if not other:
            # H terminus; implicit/explicit hydrogen is allowed only if ordinary.
            if atom.GetTotalNumHs() != 1:
                return None
            continue
        idx_other = other[0]
        neighbor = mol.GetAtomWithIdx(idx_other)
        if neighbor.GetAtomicNum() == 6:
            double_o = [n.GetIdx() for n in neighbor.GetNeighbors()
                        if n.GetAtomicNum() == 8
                        and mol.GetBondBetweenAtoms(idx_other, n.GetIdx()).GetBondTypeAsDouble() == 2]
            if double_o:
                result = _simple_acyl(mol, idx_other, idx, carbon_atoms)
                if result is None:
                    return None
                owned, count, oxygen = result
                carbonyls.append(idx_other)
                oxygens.append(oxygen)
                components.append(owned)
                counts.append(count)
                continue
            # A simple straight-chain alkyl terminus is an organyl component
            # on the homogeneous chalcogen path (P-68.4.1.3). Its exact
            # attachment and atoms remain mapped; branching/rings decline.
            if neighbor.GetAtomicNum() == 6 and idx_other in carbon_atoms:
                group = {idx_other}
                pending = [idx_other]
                valid = _single(mol, idx, idx_other)
                while pending and valid:
                    current = pending.pop()
                    ca = mol.GetAtomWithIdx(current)
                    if ca.IsInRing() or ca.GetIsAromatic() or ca.GetTotalValence() != 4:
                        valid = False
                        break
                    for cn in ca.GetNeighbors():
                        ni = cn.GetIdx()
                        if ni == idx:
                            continue
                        if cn.GetAtomicNum() != 6:
                            valid = False
                            break
                        if ni not in group:
                            if not _single(mol, current, ni):
                                valid = False
                                break
                            group.add(ni)
                            pending.append(ni)
                edge_count = sum(1 for b in iter_bonds(mol)
                                 if b.GetBeginAtomIdx() in group
                                 and b.GetEndAtomIdx() in group)
                if valid and edge_count == len(group) - 1:
                    degree = {i: sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                                     if n.GetAtomicNum() == 6 and n.GetIdx() in group)
                              for i in group}
                    if (max(degree.values(), default=0) <= 2
                            and degree.get(idx_other, 0) <= 1
                            and sum(d == 1 for d in degree.values())
                            == (0 if len(group) == 1 else 2)):
                        root = ROOTS.get(len(group))
                        if root:
                            terminal_organyl_atoms = frozenset(group)
                            terminal_organyl_name = root + "yl"
                            continue
        return None
    if len(carbonyls) not in {1, 2}:
        return None
    if len(carbonyls) == 1:
        if not terminal_organyl_atoms:
            # The opposite path terminus has its single hydrogen implicitly.
            acyl_end = next(n.GetIdx() for n in mol.GetAtomWithIdx(carbonyls[0]).GetNeighbors()
                            if n.GetIdx() in chain_atoms and n.GetAtomicNum() == element
                            and _single(mol, carbonyls[0], n.GetIdx()))
            opposite = next(i for i in terminal if i != acyl_end)
            if mol.GetAtomWithIdx(opposite).GetTotalNumHs() != 1:
                return None
    elif terminal_organyl_atoms:
        return None
    owned = set(chain_atoms)
    for component in components:
        if owned & component:
            return None
        owned.update(component)
    if terminal_organyl_atoms:
        if terminal_organyl_atoms & owned:
            return None
        owned.update(terminal_organyl_atoms)
    if owned != heavy:
        return None
    if len(carbonyls) == 2 and counts[0] == counts[1]:
        # Equal acyl identities require equal, unlabelled complete carbon units;
        # atom identities are compared through graph isomorphism downstream.
        pass
    return PseudoketoneSite(element, chain, tuple(carbonyls), tuple(oxygens),
                            tuple(components), tuple(counts), terminal_organyl_atoms,
                            terminal_organyl_name)


def _parent_name(carbon_count):
    root = ROOTS.get(carbon_count)
    return f"{root}an-1-one" if root and carbon_count >= 2 else None


def _chain_words(element, count):
    multiplier = _PREFIX_MULT.get(count)
    if multiplier is None:
        return None
    _, hydride, ylyl = _CHALCOGEN[element]
    return multiplier + hydride, multiplier + ylyl, multiplier + hydride + "diyl"


def _same_acyl_graph(mol, left, right):
    """Compare complete acyl arms after deleting the common chalcogen path."""
    from rdkit import Chem

    a, b = Chem.Mol(mol), Chem.Mol(mol)
    # Compare mapped-index independent arm SMILES after removing everything
    # outside the carbonyl C/O and its linear alkyl carbon path.
    def signature(center):
        carbon = mol.GetAtomWithIdx(center)
        seen, stack = {center}, [center]
        while stack:
            i = stack.pop()
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                j = n.GetIdx()
                if j in seen or n.GetAtomicNum() not in {6, 8}:
                    continue
                if n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 2:
                    continue
                seen.add(j); stack.append(j)
        sub = Chem.PathToSubmol(mol, [bond.GetIdx() for bond in iter_bonds(mol)
            if bond.GetBeginAtomIdx() in seen and bond.GetEndAtomIdx() in seen])
        return Chem.MolToSmiles(sub, canonical=True, isomericSmiles=True)
    return signature(left) == signature(right)


def chalcogen_chain_pseudoketone_decision(mol, Chem):
    """Return a complete source-backed MethodDecision, or decline."""
    site = recognize_chalcogen_chain_pseudoketone(mol)
    if site is None:
        return _single_acyl_branched_decision(mol)
    words = _chain_words(site.element, len(site.chain))
    if words is None:
        return None
    _, suffix, linker = words
    parent_names = [_parent_name(n) for n in site.alkyl_counts]
    if any(name is None for name in parent_names):
        return None
    locants, roles = [], []
    centers = frozenset(site.carbonyls)
    if len(site.carbonyls) == 1:
        carbonyl = site.carbonyls[0]
        acyl_order = _carbonyl_order(mol, carbonyl, site.carbon_components[0])
        chain_order = _chain_order_from_carbonyl(mol, site.chain, carbonyl)
        if not acyl_order or not chain_order:
            return None
        prefix = suffix
        if site.terminal_organyl_name is not None:
            prefix = f"{site.terminal_organyl_name}{suffix}"
            prefix = f"({prefix})"
        name = f"1-{prefix}{parent_names[0]}"
        roles.append(("carbonyl_parent", site.carbon_components[0],
                      tuple((idx, str(pos)) for pos, idx in enumerate(
                          acyl_order, 1))))
        roles.append(("homogeneous_chalcogen_chain", frozenset(site.chain),
                      tuple((idx, str(pos)) for pos, idx in enumerate(chain_order, 1))))
        if site.terminal_organyl_atoms:
            roles.append(("terminal_organyl", site.terminal_organyl_atoms, ()))
        locants.append((carbonyl, "1"))
    elif _same_acyl_graph(mol, *site.carbonyls):
        left, right = site.carbonyls
        arm = site.carbon_components[0]
        arm_order = _carbonyl_order(mol, left, arm)
        right_order = _carbonyl_order(mol, right, site.carbon_components[1])
        chain_order = _chain_order_from_carbonyl(mol, site.chain, left)
        if not arm_order or not right_order or not chain_order:
            return None
        name = f"1,1′-{linker}di({parent_names[0]})"
        method = "multiplicative"
        roles.extend((
            ("carbonyl_parent:1", site.carbon_components[0], tuple(
                (idx, str(pos)) for pos, idx in enumerate(arm_order, 1))),
            ("carbonyl_parent:2", site.carbon_components[1], tuple(
                (idx, str(pos) + "′") for pos, idx in enumerate(right_order, 1))),
            ("homogeneous_chalcogen_linker", frozenset(site.chain), tuple(
                (idx, str(pos)) for pos, idx in enumerate(chain_order, 1))),
        ))
        locants.extend(((left, "1"), (right, "1′")))
    else:
        # For these linear acyclic carbonyl parents, P-44's parent-size
        # criterion selects the longer arm. Equal-length unlike arms remain
        # unresolved and decline.
        if site.alkyl_counts[0] == site.alkyl_counts[1]:
            return None
        ordered = sorted(zip(site.alkyl_counts, site.carbonyls,
                             site.carbon_components), key=lambda row: row[0])
        lower_carbonyl, main_carbonyl = ordered[0][1], ordered[1][1]
        main_component, lower_component = ordered[1][2], ordered[0][2]
        main_order = _carbonyl_order(mol, main_carbonyl, main_component)
        lower_order = _carbonyl_order(mol, lower_carbonyl, lower_component)
        chain_order = _chain_order_from_carbonyl(mol, site.chain, main_carbonyl)
        lower_name = _acyl_prefix(ordered[0][0])
        if not main_order or not lower_order or not chain_order or lower_name is None:
            return None
        name = f"1-({lower_name}{suffix}){_parent_name(ordered[1][0])}"
        roles.extend((
            ("preferred_carbonyl_parent", main_component, tuple(
                (idx, str(pos)) for pos, idx in enumerate(main_order, 1))),
            ("secondary_acyl_component", lower_component, tuple(
                (idx, str(pos)) for pos, idx in enumerate(lower_order, 1))),
            ("homogeneous_chalcogen_chain", frozenset(site.chain), tuple(
                (idx, str(pos)) for pos, idx in enumerate(chain_order, 1))),
        ))
        method = "substitutive"
        locants.append((main_carbonyl, "1"))
    role_sets = [set(atoms) for _, atoms, _ in roles]
    all_heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if (any(role_sets[i] & role_sets[j] for i in range(len(role_sets))
            for j in range(i + 1, len(role_sets)))
            or set().union(*role_sets) != all_heavy
            or any({i for i, _ in mapping} - set(atoms)
                   for _, atoms, mapping in roles)):
        return None
    if len(site.carbonyls) == 1:
        method = "substitutive"
    return MethodDecision(name, method, centers, frozenset(all_heavy), RULE,
                          tuple(locants), component_roles=tuple(roles))


def _acyl_prefix(carbon_count):
    root = ROOTS.get(carbon_count)
    if root is None:
        return None
    return "acetyl" if carbon_count == 2 else root + "anoyl"


def _chain_order_from_carbonyl(mol, chain, carbonyl):
    attached = [i for i in (chain[0], chain[-1])
                if mol.GetBondBetweenAtoms(i, carbonyl) is not None]
    if len(attached) != 1:
        return ()
    return tuple(chain if chain[0] == attached[0] else reversed(chain))


def _carbonyl_order(mol, carbonyl, component):
    """Return the unique C1-to-terminal carbon path for an admitted acyl arm."""
    carbonyl_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                          if n.GetAtomicNum() == 6 and n.GetIdx() in component]
    if len(carbonyl_neighbors) != 1:
        return ()
    ordered, previous, current = [carbonyl], carbonyl, carbonyl_neighbors[0]
    while True:
        ordered.append(current)
        choices = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                   if n.GetAtomicNum() == 6 and n.GetIdx() in component and n.GetIdx() != previous]
        if not choices:
            break
        if len(choices) != 1:
            return ()
        previous, current = current, choices[0]
    return tuple(ordered) if len(ordered) == len([i for i in component
                                                   if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]) else ()


def _saturated_carbon_tree(mol, root, boundary):
    """Prove a complete, saturated carbon component behind one cut bond."""
    seen, pending = {root}, [root]
    while pending:
        idx = pending.pop()
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetIsAromatic()
                or atom.GetTotalValence() != 4):
            return None
        for neighbor in atom.GetNeighbors():
            ni = neighbor.GetIdx()
            if ni == boundary or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6 or not _single(mol, idx, ni):
                return None
            if ni not in seen:
                seen.add(ni)
                pending.append(ni)
    edges = sum(1 for bond in iter_bonds(mol)
                if bond.GetBeginAtomIdx() in seen and bond.GetEndAtomIdx() in seen)
    return frozenset(seen) if edges == len(seen) - 1 else None


def _methyl_branched_acyl_parent(mol, carbonyl, carbons):
    """Choose the longest carbonyl-rooted chain with only methyl branches.

    P-44.3 and P-14.4 choose the longest chain, then the lowest set of
    substituent locants. More complicated off-chain trees decline rather than
    being ranked or named by this intentionally bounded grammar.
    """
    root = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
            if n.GetIdx() in carbons]
    if len(root) != 1:
        return None
    paths, stack = [], [(carbonyl, root[0], (carbonyl,))]
    while stack:
        previous, current, prefix = stack.pop()
        path = prefix + (current,)
        children = [n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
                    if n.GetIdx() in carbons and n.GetIdx() != previous]
        if children:
            stack.extend((current, child, path) for child in children)
        else:
            paths.append(path)
    if not paths:
        return None
    longest = max(map(len, paths))
    if _parent_name(longest) is None:
        return None
    possibilities = []
    for path in paths:
        if len(path) != longest:
            continue
        path_set = set(path)
        locants = []
        for position, idx in enumerate(path, 1):
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                ni = neighbor.GetIdx()
                if ni not in carbons or ni in path_set:
                    continue
                # An off-chain branch must consist of precisely one methyl.
                if any(n.GetAtomicNum() == 6 and n.GetIdx() in carbons
                       and n.GetIdx() != idx for n in neighbor.GetNeighbors()):
                    break
                locants.append(position)
            else:
                continue
            break
        else:
            if len(locants) <= 1 or len(locants) in MULT:
                possibilities.append((tuple(sorted(locants)), path))
    if not possibilities:
        return None
    locants, path = min(possibilities, key=lambda item: item[0])
    return path, locants


def _simple_terminal_organyl(mol, root, terminal):
    """Admit a terminal linear alkyl or the retained tert-butyl prefix."""
    if not _single(mol, root, terminal):
        return None
    atoms = _saturated_carbon_tree(mol, root, terminal)
    if not atoms:
        return None
    degrees = {i: sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())
               for i in atoms}
    if (len(atoms) == 4 and degrees[root] == 3
            and all(degrees[i] == 1 for i in atoms if i != root)):
        return atoms, "tert-butyl"
    if max(degrees.values(), default=0) > 2 or degrees[root] > 1:
        return None
    root_name = ROOTS.get(len(atoms))
    return (atoms, root_name + "yl") if root_name else None


def _single_acyl_branched_decision(mol):
    """Handle one acyl terminus when a proved simple carbon tree branches.

    P-68.4.1.3 requires a pseudoketone even when the acyl carbon component or
    the other terminal organyl group is branched. This narrowly admits one
    carbonyl, a homogeneous neutral chain and a methyl-branched saturated
    acyl parent; other functionalities, stereochemistry, and multiple acyl
    arms are deliberately left to their own methods.
    """
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           or int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    candidates = []
    for element in _CHALCOGEN:
        atoms = {a.GetIdx() for a in iter_atoms(mol)
                 if a.GetAtomicNum() == element
                 and not any(b.GetBondTypeAsDouble() > 1 for b in a.GetBonds())}
        chain = _path(mol, atoms) if len(atoms) >= 3 else ()
        if chain:
            candidates.append((element, chain))
    if len(candidates) != 1:
        return None
    element, chain = candidates[0]
    words = _chain_words(element, len(chain))
    if words is None or any(a.GetAtomicNum() not in {1, 6, 8, element}
                            for a in iter_atoms(mol)):
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in chain):
        return None
    if any(not _single(mol, a, b) for a, b in zip(chain, chain[1:])):
        return None
    chain_set = frozenset(chain)
    for idx in chain[1:-1]:
        if sum(n.GetAtomicNum() > 1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()) != 2:
            return None
    ends = []
    for idx in (chain[0], chain[-1]):
        extras = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                  if n.GetAtomicNum() > 1 and n.GetIdx() not in chain_set]
        if len(extras) > 1:
            return None
        if not extras and mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True) != 1:
            return None
        ends.append((idx, extras[0] if extras else None))
    acyls = []
    for terminal, other in ends:
        if other is None or mol.GetAtomWithIdx(other).GetAtomicNum() != 6:
            continue
        atom = mol.GetAtomWithIdx(other)
        oxo = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(other, n.GetIdx()).GetBondTypeAsDouble() == 2]
        if oxo:
            acyls.append((terminal, other, oxo))
    if len(acyls) != 1:
        return None
    acyl_end, carbonyl, oxo = acyls[0]
    heavy_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                       if n.GetAtomicNum() > 1]
    if len(oxo) != 1 or len(heavy_neighbors) != 3 or not _single(mol, acyl_end, carbonyl):
        return None
    carbon_roots = [i for i in heavy_neighbors if i not in {acyl_end, oxo[0]}]
    if len(carbon_roots) != 1 or not _single(mol, carbonyl, carbon_roots[0]):
        return None
    carbon_tree = _saturated_carbon_tree(mol, carbon_roots[0], carbonyl)
    if not carbon_tree:
        return None
    parent = _methyl_branched_acyl_parent(mol, carbonyl, carbon_tree)
    if parent is None:
        return None
    parent_path, methyl_locants = parent
    other_end, terminal_root = next(row for row in ends if row[0] != acyl_end)
    terminal_atoms, terminal_name = frozenset(), None
    if terminal_root is not None:
        organyl = _simple_terminal_organyl(mol, terminal_root, other_end)
        if organyl is None:
            return None
        terminal_atoms, terminal_name = organyl
    # Do not duplicate the straight-chain route or its established output.
    if not methyl_locants and terminal_name != "tert-butyl":
        return None
    carbon_component = carbon_tree | {carbonyl, oxo[0]}
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if (chain_set & carbon_component or chain_set & terminal_atoms
            or carbon_component & terminal_atoms
            or chain_set | carbon_component | terminal_atoms != all_heavy):
        return None
    _, suffix, _ = words
    chain_name = (terminal_name or "") + suffix
    chain_piece = f"1-({chain_name})" if terminal_name else f"1-{chain_name}"
    pieces = [(chain_name, chain_piece)]
    if methyl_locants:
        multiplier = MULT[len(methyl_locants)] if len(methyl_locants) > 1 else ""
        pieces.append(("methyl", f"{','.join(map(str, methyl_locants))}-{multiplier}methyl"))
    from .formatting import prefix_alphabetization_key
    pieces.sort(key=lambda pair: prefix_alphabetization_key(pair[0]))
    name = "-".join(piece for _, piece in pieces) + _parent_name(len(parent_path))
    ordered_chain = chain if chain[0] == acyl_end else tuple(reversed(chain))
    roles = [
        ("carbonyl_parent", carbon_component,
         tuple((idx, str(pos)) for pos, idx in enumerate(parent_path, 1))),
        ("homogeneous_chalcogen_chain", chain_set,
         tuple((idx, str(pos)) for pos, idx in enumerate(ordered_chain, 1))),
    ]
    if terminal_atoms:
        roles.append(("terminal_organyl", terminal_atoms, ()))
    return MethodDecision(name, "substitutive", frozenset({carbonyl}), all_heavy,
                          RULE, ((carbonyl, "1"),), component_roles=tuple(roles))

