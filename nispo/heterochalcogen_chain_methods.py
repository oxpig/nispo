"""Bounded mapped P-68.4.2.1 heterogeneous chalcogen parent hydrides.

Admits neutral, ordinary-divalent alternating O/S/Se/Te chains of the form
A-B-A-(B-A)n (three or more atoms), where A is lower in the source seniority
order than B, and one or two carbon-only organyl groups at A termini.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT
from .naming_methods import MethodDecision


RULE = "P-41/P-44/P-45/P-51.3/P-16.3.2/P-16.3.3/P-16.3.4/P-16.3.5/P-16.5.1.2/P-16.5.1.3.1/P-68.4.0/P-68.4.2.1/P-21.2.3.1/P-68.4.2.4/P-29.3/P-82/P-91"
_ELEMENTS = {
    8: ("oxygen", "oxa"), 16: ("sulfur", "thia"),
    34: ("selenium", "selena"), 52: ("tellurium", "tellura"),
}
# Group 16 seniority from P-21.2.3.1: O > S > Se > Te.
_SENIORITY = {8: 0, 16: 1, 34: 2, 52: 3}


@dataclass(frozen=True, slots=True)
class HeterochalcogenChainSite:
    chain_atoms: tuple[int, ...]
    terminal_element: int
    internal_element: int
    organyl_roots: tuple[int, ...]
    organyl_atoms: tuple[frozenset[int], ...]


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _connected_component(mol, start, allowed):
    seen, pending = {start}, [start]
    while pending:
        current = pending.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx in allowed and idx not in seen:
                seen.add(idx)
                pending.append(idx)
    return frozenset(seen)


def _ordered_path(mol, atoms):
    adjacency = {
        idx: sorted(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if n.GetIdx() in atoms)
        for idx in atoms
    }
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return ()
    ends = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(ends) != 2:
        return ()
    path, previous, current = [], None, ends[0]
    while True:
        path.append(current)
        choices = [idx for idx in adjacency[current] if idx != previous]
        if not choices:
            break
        if len(choices) != 1 or choices[0] in path:
            return ()
        previous, current = current, choices[0]
    return tuple(path) if len(path) == len(atoms) and path[-1] == ends[1] else ()


def _organyl_component(mol, root, attachment, allowed_carbons):
    """Return a bounded carbon tree or unsubstituted phenyl branch."""
    if not _single(mol, root, attachment):
        return None
    atoms = _connected_component(mol, root, allowed_carbons)
    if not atoms:
        return None
    links = {(i, n.GetIdx()) for i in atoms
             for n in mol.GetAtomWithIdx(i).GetNeighbors()
             if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
    if links != {(root, attachment)}:
        return None
    ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()
                 if set(ring) & atoms]
        if (len(rings) != 1 or len(atoms) != 6 or len(rings[0]) != 6
                or set(rings[0]) != set(atoms)
                or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
                or any(not mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                       for a, b in zip(rings[0], rings[0][1:] + rings[0][:1]))):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return None
        if any(not _single(mol, a, b) for a in atoms
               for b in (n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
               if b in atoms):
            return None
        edge_count = sum(1 for b in iter_bonds(mol)
                         if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms)
        if edge_count != len(atoms) - 1:
            return None
    return atoms


def recognize_heterochalcogen_chain(mol):
    """Recognize one exact P-21.2.3.1 alternating chalcogen chain."""
    if (mol.GetNumAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))
            or any(a.GetAtomicNum() == 1 and a.GetIsotope()
                   for a in iter_atoms(mol))):
        return None
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    chalcogens = {a.GetIdx() for a in iter_atoms(mol)
                  if a.GetAtomicNum() in _ELEMENTS}
    carbons = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    if not chalcogens or chalcogens | carbons != heavy:
        return None
    if len({mol.GetAtomWithIdx(i).GetAtomicNum() for i in chalcogens}) != 2:
        return None
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag())
           or mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetIsAromatic()
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2
           for i in chalcogens):
        return None
    if _connected_component(mol, min(chalcogens), chalcogens) != frozenset(chalcogens):
        return None
    chain = _ordered_path(mol, chalcogens)
    if len(chain) < 3 or len(chain) % 2 == 0:
        return None
    if any(not _single(mol, a, b) for a, b in zip(chain, chain[1:])):
        return None
    sequence = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)
    terminal_element, internal_element = sequence[0], sequence[1]
    if (_SENIORITY[terminal_element] <= _SENIORITY[internal_element]
            or sequence[-1] != terminal_element
            or any(z != (terminal_element if i % 2 == 0 else internal_element)
                   for i, z in enumerate(sequence))):
        return None

    terminal_set = {chain[0], chain[-1]}
    roots, components, used_carbons = [], [], set()
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        chain_neighbors = [n for n in neighbors if n in chalcogens]
        carbon_neighbors = [n for n in neighbors if n in carbons]
        if any(n not in chalcogens | carbons for n in neighbors):
            return None
        if idx in terminal_set:
            if len(chain_neighbors) != 1 or len(carbon_neighbors) > 1:
                return None
            if carbon_neighbors:
                root = carbon_neighbors[0]
                component = _organyl_component(mol, root, idx, carbons - used_carbons)
                if component is None or component & used_carbons:
                    return None
                roots.append(root)
                components.append(component)
                used_carbons.update(component)
        elif len(chain_neighbors) != 2 or carbon_neighbors:
            return None
    if used_carbons != carbons or not 1 <= len(roots) <= 2:
        return None
    return HeterochalcogenChainSite(
        chain, terminal_element, internal_element, tuple(roots), tuple(components)
    )


def _parent_hydride_name(chain_length, terminal_element, internal_element):
    """Apply the P-21.2.3.1 term recipe with its specified vowel elision."""
    count = (chain_length + 1) // 2
    multiplier = MULT.get(count)
    if not multiplier:
        return ""
    terminal_term = _ELEMENTS[terminal_element][1]
    internal_term = _ELEMENTS[internal_element][1]
    text = multiplier + terminal_term
    for term in (internal_term, "ane"):
        if text.endswith("a") and term[:1].lower() in "aeiou":
            text = text[:-1]
        text += term
    return text


def _organyl_nature(mol, prefix, atoms):
    """Classify the mapped carbon graph for the P-16 multiplier rules."""
    if prefix in {"phenyl", "tert-butyl"}:
        return "simple"
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
             for bond in iter_bonds(mol)
             if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms]
    degrees = {idx: sum(idx in edge for edge in edges) for idx in atoms}
    # An unbranched alkyl skeleton remains a simple prefix even where the
    # attachment atom is internal (e.g. propan-2-yl); P-16.3.4(a) then
    # supplies its locant parentheses. A branched carbon skeleton is a
    # compound prefix and takes bis/tris, etc.
    if len(edges) == len(atoms) - 1 and all(degree <= 2 for degree in degrees.values()):
        return "simple"
    return "compound"


def _organyl_needs_enclosure(prefix, nature):
    """Whether a complete organyl prefix needs enclosing marks in citation."""
    return (nature == "compound" or prefix.startswith(("(", "["))
            or ("-" in prefix and prefix != "tert-butyl"))


def _prefix_text(mol, prefix_rows, chain, need_locants):
    from .formatting import (enclose_name_unit, format_prefixes,
                             prefix_alphabetization_key)

    location = {idx: pos for pos, idx in enumerate(chain, 1)}
    if need_locants:
        if len(prefix_rows) == 2 and prefix_rows[0][0] == prefix_rows[1][0]:
            first, atoms, _ = prefix_rows[0]
            nature = _organyl_nature(mol, first, atoms)
            if nature != _organyl_nature(mol, prefix_rows[1][0], prefix_rows[1][1]):
                return ""
            from .substituent_multiplication import multiply_substituent
            operation = multiply_substituent(first, 2, nature=nature)
            if operation is None:
                return ""
            locants = sorted(location[attached] for _name, _atoms, attached in prefix_rows)
            return ",".join(str(locant) for locant in locants) + "-" + operation.text
        return format_prefixes([
            (str(location[attached]), prefix)
            for prefix, _atoms, attached in prefix_rows
        ], hierarchical_enclosures=True)
    if len(prefix_rows) == 1:
        prefix, atoms, _attached = prefix_rows[0]
        if _organyl_needs_enclosure(
                prefix, _organyl_nature(mol, prefix, atoms)):
            return enclose_name_unit(prefix)
        return prefix
    if len(prefix_rows) != 2:
        return ""
    first, second = prefix_rows[0][0], prefix_rows[1][0]
    ordered_rows = sorted(prefix_rows, key=lambda row: prefix_alphabetization_key(row[0]))
    first, first_atoms, _ = ordered_rows[0]
    second, second_atoms, _ = ordered_rows[1]
    first_nature = _organyl_nature(mol, first, first_atoms)
    second_nature = _organyl_nature(mol, second, second_atoms)
    if first == second:
        if first_nature != second_nature:
            return ""
        from .substituent_multiplication import multiply_substituent
        operation = multiply_substituent(first, 2, nature=first_nature)
        return operation.text if operation else ""
    first_text = (enclose_name_unit(first)
                  if _organyl_needs_enclosure(first, first_nature) else first)
    return first_text + enclose_name_unit(second)


def _prefix_locants_required(mol, chain, prefix_rows):
    if not prefix_rows:
        return False
    core_pattern = tuple(mol.GetAtomWithIdx(i).GetIsotope() for i in chain)
    core_asymmetric = core_pattern != tuple(reversed(core_pattern))
    if not core_asymmetric:
        return False
    if len(prefix_rows) == 1:
        return True
    if len(prefix_rows) != 2:
        return True
    arm_facts = []
    for name, atoms, _endpoint in prefix_rows:
        branch_labels = tuple(sorted(
            (mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(i).GetIsotope())
            for i in atoms if mol.GetAtomWithIdx(i).GetIsotope()
        ))
        arm_facts.append((name, branch_labels))
    return arm_facts[0] != arm_facts[1]


def heterochalcogen_chain_decision(mol, Chem):
    """Return a fully mapped substitutive decision within the admitted scope."""
    site = recognize_heterochalcogen_chain(mol)
    if site is None:
        return None
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .formatting import prefix_alphabetization_key
    from .parent_annotations import isotope_numbering_key

    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    prefix_rows = []
    for root, expected_atoms in zip(site.organyl_roots, site.organyl_atoms, strict=True):
        attached = next(n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
                        if n.GetIdx() in site.chain_atoms)
        prefix, atoms = mapped_branch(mol, Chem, root, attached)
        if not prefix or atoms != expected_atoms:
            return None
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
            return None
        prefix_rows.append((prefix, atoms, attached))

    chain = tuple(site.chain_atoms)
    reverse = tuple(reversed(chain))
    # The preselected a[ba]n parent fixes the heteroatom sequence in either
    # direction. Prefix citation resolves different unmodified arms first.
    attached = {chain[0]: [], chain[-1]: []}
    for prefix, _atoms, endpoint in prefix_rows:
        attached[endpoint].append(prefix)
    left = tuple(sorted(attached[chain[0]], key=prefix_alphabetization_key))
    right = tuple(sorted(attached[chain[-1]], key=prefix_alphabetization_key))
    if len(left) != len(right):
        chain = chain if left else reverse
    elif left and right:
        left_key = tuple(prefix_alphabetization_key(x)[:3] for x in left)
        right_key = tuple(prefix_alphabetization_key(x)[:3] for x in right)
        if left_key != right_key:
            chain = chain if left_key < right_key else reverse
        else:
            mapping = {idx: str(pos) for pos, idx in enumerate(chain, 1)}
            reverse_mapping = {idx: str(pos) for pos, idx in enumerate(reverse, 1)}
            forward_key = isotope_numbering_key(
                mol, mapping, include_substituent_groups=True
            )
            reverse_key = isotope_numbering_key(
                mol, reverse_mapping, include_substituent_groups=True
            )
            if forward_key is None or reverse_key is None:
                return None
            if reverse_key < forward_key:
                chain = reverse
    else:
        # Both ends carry no organyl only; that is outside this producer.
        return None

    parent = _parent_hydride_name(len(chain), site.terminal_element,
                                  site.internal_element)
    if not parent:
        return None
    chain_locs = tuple((idx, str(pos)) for pos, idx in enumerate(chain, 1))
    core_parent = decorate_component(
        mol, Chem,
        MappedComponent(parent=parent, atom_locants=chain_locs,
                        atoms=frozenset(chain), systematic_parent=True),
    )
    if not core_parent:
        return None
    prefix = _prefix_text(mol, prefix_rows, chain,
                          _prefix_locants_required(mol, chain, prefix_rows))
    if not prefix:
        return None

    roles = [("heterochalcogen_parent_hydride", frozenset(chain), chain_locs)]
    roles.extend((f"organyl:{n}", atoms, ())
                 for n, (_text, atoms, _endpoint) in enumerate(prefix_rows, 1))
    role_sets = [set(atoms) for _name, atoms, _locants in roles]
    if (any(role_sets[i] & role_sets[j]
            for i in range(len(role_sets)) for j in range(i + 1, len(role_sets)))
            or set().union(*role_sets) != set(all_heavy)
            or any({idx for idx, _loc in locants} - set(atoms)
                   for _name, atoms, locants in roles)):
        return None
    return MethodDecision(
        prefix + core_parent, "substitutive", frozenset(), all_heavy, RULE,
        chain_locs, component_roles=tuple(roles),
    )
