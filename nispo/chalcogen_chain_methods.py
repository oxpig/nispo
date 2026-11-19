"""Bounded mapped parent hydrides for homogeneous chalcogen chains (P-68.4.1).

This producer recognizes one neutral, unbranched chain of at least three
identical divalent O/S/Se/Te atoms. Up to two terminal carbon organyl groups
are named substitutively on that parent hydride. Carbon groups are rendered
from the original mapped graph; unsupported functionality and annotations on
the chalcogen parent decline rather than being dropped.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT
from .naming_methods import MethodDecision


RULE = "P-41/P-44/P-45/P-51.3/P-68.4.0/P-68.4.1.1/P-21.2.2/P-29.3/P-82/P-91"
_ELEMENTS = {8: ("oxygen", "oxidane"), 16: ("sulfur", "sulfane"),
             34: ("selenium", "selane"), 52: ("tellurium", "tellane")}


@dataclass(frozen=True, slots=True)
class ChalcogenChainSite:
    element: int
    chain_atoms: tuple[int, ...]
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
    if len(ends) != 2 or len(atoms) < 3:
        return ()
    path, previous, current = [], None, ends[0]
    while True:
        path.append(current)
        choices = [idx for idx in adjacency[current] if idx != previous]
        if not choices:
            break
        if len(choices) != 1:
            return ()
        previous, current = current, choices[0]
        if current in path:
            return ()
    if len(path) != len(atoms) or path[-1] != ends[1]:
        return ()
    return tuple(path)


def _organyl_component(mol, root, attachment, allowed_carbons):
    """Validate and return a simple carbon-only organyl attached at one end."""
    if not _single(mol, root, attachment):
        return None
    atoms = _connected_component(mol, root, allowed_carbons)
    if not atoms or any(mol.GetAtomWithIdx(i).GetIsAromatic() and
                        not mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return None
    links = [(i, n.GetIdx()) for i in atoms
             for n in mol.GetAtomWithIdx(i).GetNeighbors()
             if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms]
    if links != [(root, attachment)] and set(links) != {(root, attachment)}:
        return None
    # P-29 organyl scope: a saturated carbon tree or a single unsubstituted
    # phenyl group. Other cyclic/aromatic groups need their own parent proof.
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
        edge_count = sum(1 for bond in iter_bonds(mol)
                         if bond.GetBeginAtomIdx() in atoms
                         and bond.GetEndAtomIdx() in atoms)
        if edge_count != len(atoms) - 1:
            return None
    return atoms


def recognize_chalcogen_chain(mol):
    """Recognize exactly one P-68.4.1 homogeneous chain and terminal organyls."""
    if (mol.GetNumAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))
            or any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol))):
        return None
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    chalcogens = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() in _ELEMENTS}
    carbons = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    if not chalcogens or not carbons or chalcogens | carbons != heavy:
        return None
    elements = {mol.GetAtomWithIdx(i).GetAtomicNum() for i in chalcogens}
    if len(elements) != 1:
        return None
    element = next(iter(elements))
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag())
           or mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetIsAromatic()
           or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2
           for i in chalcogens):
        return None
    # The whole chalcogen set must be one contiguous homogeneous simple path.
    start = min(chalcogens)
    if _connected_component(mol, start, chalcogens) != frozenset(chalcogens):
        return None
    chain = _ordered_path(mol, chalcogens)
    if not chain or any(not _single(mol, a, b) for a, b in zip(chain, chain[1:])):
        return None
    terminal_set = {chain[0], chain[-1]}
    roots, components = [], []
    used_carbons = set()
    for position, idx in enumerate(chain):
        atom = mol.GetAtomWithIdx(idx)
        e_neighbors = [n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == element]
        c_neighbors = [n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == 6]
        if any(n.GetAtomicNum() not in {1, 6, element}
               for n in atom.GetNeighbors()):
            return None
        if idx in terminal_set:
            if len(e_neighbors) != 1 or len(c_neighbors) > 1:
                return None
            if c_neighbors:
                root = c_neighbors[0]
                component = _organyl_component(mol, root, idx, carbons - used_carbons)
                if component is None or component & used_carbons:
                    return None
                roots.append(root)
                components.append(component)
                used_carbons.update(component)
        elif len(e_neighbors) != 2 or c_neighbors:
            return None
    if used_carbons != carbons or not roots:
        return None
    # Every carbon must be owned by one of the terminal organyl groups.
    return ChalcogenChainSite(element, chain, tuple(roots), tuple(components))


def chalcogen_chain_decision(mol, Chem):
    """Build a fully mapped substitutive decision or decline outside scope."""
    site = recognize_chalcogen_chain(mol)
    if site is None:
        return None
    from .component_mapping import MappedComponent, decorate_component, mapped_branch

    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    prefix_rows = []
    for root, expected_atoms in zip(site.organyl_roots, site.organyl_atoms, strict=True):
        attached = next(n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
                        if n.GetAtomicNum() in _ELEMENTS)
        prefix, atoms = mapped_branch(mol, Chem, root, attached)
        if not prefix or atoms != expected_atoms:
            return None
        # Keep the prefix grammar bounded to simple alkyl and phenyl organyls.
        base = prefix
        if (any(ch in base for ch in "{}[]")
                or any(a.GetAtomicNum() != 6 for i in atoms
                       for a in [mol.GetAtomWithIdx(i)])):
            return None
        prefix_rows.append((prefix, atoms, attached))
    base_parent = _ELEMENTS[site.element][1]
    multiplicative = MULT.get(len(site.chain_atoms))
    if not multiplicative:
        return None
    parent = multiplicative + base_parent
    from .formatting import enclose_name_unit, prefix_alphabetization_key

    chain = _number_isotopic_chain(mol, site.chain_atoms, prefix_rows,
                                   prefix_alphabetization_key)
    if chain is None:
        return None
    chain_locs = tuple((idx, str(pos)) for pos, idx in enumerate(chain, 1))
    core_parent = decorate_component(
        mol, Chem,
        MappedComponent(parent=parent, atom_locants=chain_locs,
                        atoms=frozenset(chain), systematic_parent=True),
    )
    if not core_parent:
        return None

    prefix_locants_required = _organyl_prefix_locants_required(
        mol, chain, prefix_rows
    )
    if len(prefix_rows) == 1:
        prefix, _atoms, attached = prefix_rows[0]
        if prefix_locants_required:
            from .formatting import format_prefixes
            prefix_text = format_prefixes([
                (str({idx: n for n, idx in enumerate(chain, 1)}[attached]), prefix)
            ], hierarchical_enclosures=True)
        else:
            prefix_text = _organyl_outer(prefix, enclose_name_unit)
    elif len(prefix_rows) == 2:
        left, right = (row[0] for row in prefix_rows)
        # P-14 excludes locants and isotope/stereo descriptors from the
        # alphanumerical key. It still compares the chemical prefix itself.
        ordered = sorted((left, right), key=prefix_alphabetization_key)
        if prefix_locants_required:
            from .formatting import format_prefixes
            prefix_text = format_prefixes([
                (str({idx: n for n, idx in enumerate(chain, 1)}[attached]), text)
                for text, _atoms, attached in prefix_rows
            ], hierarchical_enclosures=True)
        elif left == right:
            if _same_chemical_prefix_different_qualifiers(left, right):
                return None
            from .substituent_multiplication import multiply_substituent
            atoms = prefix_rows[0][1]
            path = all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                       for i in atoms)
            nature = 'simple' if left in {'phenyl', 'tert-butyl'} or path else 'compound'
            operation = multiply_substituent(left, 2, nature=nature)
            if operation is None:
                return None
            prefix_text = operation.text
        else:
            # Multiple independently modified forms of the same chemical
            # substituent need the P-82.2.2.1 citation ordering and enclosing
            # grammar. Until that distinction is represented explicitly, do
            # not choose an order from rendered strings.
            if _same_chemical_prefix_different_qualifiers(left, right):
                return None
            first, second = ordered
            first_text = (_organyl_outer(first, enclose_name_unit)
                          if _is_compound_organyl(_unqualify_prefix(first))
                          or _is_locanted_unsubstituted_alkyl(_unqualify_prefix(first))
                          else first)
            prefix_text = first_text + enclose_name_unit(second)
    else:
        return None
    if not prefix_text:
        return None
    roles = [("chalcogen_parent_hydride", frozenset(chain), chain_locs)]
    roles.extend((f"organyl:{n}", atoms, ())
                 for n, (_, atoms, _) in enumerate(prefix_rows, 1))
    role_atoms = [set(atoms) for _, atoms, _ in roles]
    if (any(role_atoms[i] & role_atoms[j]
            for i in range(len(role_atoms)) for j in range(i + 1, len(role_atoms)))
            or set().union(*role_atoms) != set(all_heavy)
            or any({idx for idx, _ in locants} - set(atoms)
                   for _, atoms, locants in roles)):
        return None
    name = prefix_text + core_parent
    return MethodDecision(name, "substitutive", frozenset(), all_heavy, RULE,
                         chain_locs, component_roles=tuple(roles))


def _is_locanted_unsubstituted_alkyl(name):
    """Return whether a mapped alkyl prefix is simple but has attachment locants."""
    import re
    from .constants import ROOTS
    roots = '|'.join(re.escape(root) for root in sorted(ROOTS.values(), key=len, reverse=True))
    return bool(re.fullmatch(rf"(?:{roots})an-\d+-yl", name))


def _is_compound_organyl(name):
    import re
    return bool(re.match(r"\d+(?:,\d+)*-", name)
                and not _is_locanted_unsubstituted_alkyl(name))


def _unqualify_prefix(text):
    import re
    text = re.sub(r"^\(\d+[RSZE]\)-", "", text)
    return re.sub(r"^\(\d+[A-Z][a-z]?\)", "", text)


def _organyl_outer(text, enclose_name_unit):
    """Enclose a complete prefix when its internal grammar needs it."""
    import re
    if re.match(r"^\(\d{2,}[A-Z][a-z]?\)", text):
        return text
    if (text.startswith("(") or text.startswith("[") or "-" in text
            or _is_compound_organyl(_unqualify_prefix(text))):
        return enclose_name_unit(text)
    return text


def _number_isotopic_chain(mol, chain, prefix_rows, prefix_key):
    """Choose a parent-chain direction before writing typed isotope locants.

    Ordinary terminal-prefix order is fixed first. If that leaves reversal
    equivalent, P-82.5.2 assigns the lower locants to isotope-modified atoms
    and then to the higher-mass isotope of the same element.
    """
    chain = tuple(chain)
    reverse = tuple(reversed(chain))
    ends = {chain[0], chain[-1]}
    attached = {end: [] for end in ends}
    for prefix, _atoms, chalcogen in prefix_rows:
        if chalcogen in attached:
            attached[chalcogen].append(prefix)
    endpoint_terms = []
    for end in (chain[0], chain[-1]):
        endpoint_terms.append(tuple(sorted(attached[end], key=prefix_key)))
    left, right = endpoint_terms
    if len(left) != len(right):
        return chain if len(left) else reverse
    if left and right:
        # P-14 prefix order controls unmodified numbering before isotope
        # locants (P-82.5.1). Descriptors are removed by prefix_key.
        left_key = tuple(prefix_key(text)[:3] for text in left)
        right_key = tuple(prefix_key(text)[:3] for text in right)
        if left_key != right_key:
            # P-14.4(g) assigns the lower locant to the first cited prefix;
            # P-82.5.1 leaves that ordinary numbering ahead of isotope rules.
            return chain if left_key < right_key else reverse

    def isotope_score(candidate):
        from .parent_annotations import isotope_numbering_key

        parent_map = {idx: str(pos) for pos, idx in enumerate(candidate, 1)}
        return isotope_numbering_key(
            mol, parent_map, include_substituent_groups=True)

    forward_score, reverse_score = isotope_score(chain), isotope_score(reverse)
    if forward_score is None or reverse_score is None:
        return None
    if forward_score != reverse_score:
        return chain if forward_score < reverse_score else reverse
    # Exact ties have the same isotope notation whichever end is called 1.
    return chain


def _organyl_prefix_locants_required(mol, chain, prefix_rows):
    """Require arm locants when isotope modification makes an exchange an isomer.

    P-14.3.4.4 and P-82.6.1.4 retain locants when exchanging terminal groups
    can generate an isomer. Parent isotope symmetry is tested on the actual
    mapped chain; an isotope-labelled organyl is distinguished from an
    otherwise identical unlabelled arm.
    """
    if not prefix_rows:
        return False
    # If the isotope pattern of the parent chain is not invariant under
    # reversal, the two terminal attachment sites are not interchangeable.
    isotope_pattern = tuple(mol.GetAtomWithIdx(idx).GetIsotope() for idx in chain)
    parent_asymmetric = isotope_pattern != tuple(reversed(isotope_pattern))
    arm_terms = []
    for prefix, atoms, _attached in prefix_rows:
        labels = tuple(sorted((mol.GetAtomWithIdx(i).GetAtomicNum(),
                               mol.GetAtomWithIdx(i).GetIsotope())
                              for i in atoms
                              if mol.GetAtomWithIdx(i).GetIsotope()))
        arm_terms.append((prefix, labels))
    arms_differ = len(arm_terms) > 1 and arm_terms[0] != arm_terms[1]
    return parent_asymmetric and (len(prefix_rows) == 1 or arms_differ)


def _same_chemical_prefix_different_qualifiers(left, right):
    """Conservatively identify matching branch terms with distinct qualifiers."""
    import re

    def base(text):
        return _unqualify_prefix(text)

    same_chemical_group = base(left) == base(right)
    if not same_chemical_group:
        return False
    isotope = re.compile(r"^\(\d{2,}[A-Z][a-z]?\)")
    left_iso, right_iso = isotope.match(left), isotope.match(right)
    if bool(left_iso) != bool(right_iso):
        # P-82.2.2.1 expressly orders a differently isotope-modified group
        # before its unmodified analogue; prefix_alphabetization_key does so
        # after stripping the descriptor and comparing the chemical term.
        return False
    left_qualified = base(left) != left
    right_qualified = base(right) != right
    # Repeated isotope-qualified groups, two different isotope labels, and
    # separately stereodescribed copies need a more specific P-82/P-91 model.
    return left_qualified or right_qualified
