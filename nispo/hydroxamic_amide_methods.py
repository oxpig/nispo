"""Mapped preferred-name route for bounded N-hydroxy/N-alkoxy amides.

The amide suffix construction comes from the shared complete-parent renderer;
this module adds the source-bounded P-41 gate and proves N/O branch ownership.
"""

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

RULE = (
    "P-41/P-43/P-44/P-66.1.1.1.1.1/.1.1.1.3/.1.1.3.1-.2/.1.1.3.5; P-82"
)


def _heavy(mol):
    return frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)


def _equivalent_frontier_mappings(mol, Chem, rows):
    """Accept only same-name frontier members related by a graph automorphism."""
    if not rows or len({row[0].name for row in rows}) != 1:
        return False
    if len(rows) == 1:
        return True
    reference = rows[0]
    ref_candidate, ref_construction, ref_parent, ref_embedded, ref_o, ref_n, ref_chain = reference
    ref_locants = dict(ref_construction.atom_locants)
    ref_roles = (ref_parent, ref_o, ref_n, ref_chain)
    if all(dict(row[1].atom_locants) == ref_locants
           and (row[2], row[4], row[5], row[6]) == ref_roles for row in rows[1:]):
        return True
    # Mapped alternatives with one identical name can be coalesced only when
    # a stereochemistry-preserving automorphism proves the locant and role
    # assignments equivalent on the original labelled graph.
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True,
                                      maxMatches=20000)
    for row in rows[1:]:
        construction = row[1]
        locants = dict(construction.atom_locants)
        roles = (row[2], row[4], row[5], row[6])
        matched = False
        for match in matches:
            if any(locants.get(match[idx]) != loc for idx, loc in ref_locants.items()):
                continue
            if any({match[idx] for idx in left} != right
                   for left, right in zip(ref_roles, roles)):
                continue
            matched = True
            break
        if not matched:
            return False
    return True


def _bounded_site(mol, group):
    """Return (amide N, O branch root, optional N-carbon root), else None."""
    n = mol.GetAtomWithIdx(group.ligand)
    c = mol.GetAtomWithIdx(group.center)
    if (n.GetAtomicNum() != 7 or n.GetFormalCharge() or n.GetNumRadicalElectrons()
            or n.IsInRing() or c.GetAtomicNum() != 6):
        return None
    if mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1:
        return None
    n_heavy = [a for a in n.GetNeighbors() if a.GetAtomicNum() > 1]
    oxygens = [a for a in n_heavy if a.GetAtomicNum() == 8]
    carbons = [a for a in n_heavy if a.GetAtomicNum() == 6 and a.GetIdx() != group.center]
    if (len(oxygens) != 1 or len(carbons) > 1
            or len(n_heavy) != 2 + len(carbons)):
        return None
    oxygen = oxygens[0]
    if (oxygen.GetFormalCharge() or oxygen.GetNumRadicalElectrons() or oxygen.IsInRing()
            or oxygen.GetIsAromatic()
            or mol.GetBondBetweenAtoms(n.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble() != 1):
        return None
    o_heavy = [a for a in oxygen.GetNeighbors() if a.GetAtomicNum() > 1]
    o_carbons = [a for a in o_heavy if a.GetAtomicNum() == 6 and a.GetIdx() != n.GetIdx()]
    if len(o_heavy) not in {1, 2} or len(o_carbons) != len(o_heavy) - 1:
        return None
    if any(a.GetAtomicNum() not in {6, 7, 8} for a in n_heavy + o_heavy):
        return None
    if int(n.GetChiralTag()) or int(oxygen.GetChiralTag()):
        return None
    if n.GetIsotope() or oxygen.GetIsotope():
        return None
    # Only ordinary carbon branches are admitted; the renderer must establish
    # their exact construction and annotation map below.
    return n.GetIdx(), oxygen.GetIdx(), (carbons[0].GetIdx() if carbons else None)


def hydroxamic_amide_decision(mol, Chem):
    """Build a complete P-66 substitutive decision, or return ``None``.

    Scope is one neutral ordinary carboxamide with exactly one N-bound oxygen,
    optionally O-alkylated, and at most one additional N-organyl branch. The
    amide must be the true P-41 principal group. Carbon skeleton annotations
    are left on the original molecule for the shared mapped component writer.
    """
    from .extended_inventory import selection_inventory
    from .complete_functional_parent import functional_parent_candidates
    from .component_mapping import mapped_branch
    from .parent_selection import parent_frontier

    if (mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                    for a in iter_atoms(mol))):
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() not in {1, 6, 7, 8} for a in atoms):
        return None
    # Core-element and hydrogen isotope grammar is not established here.
    if any(a.GetAtomicNum() in {1, 7, 8} and a.GetIsotope() for a in atoms):
        return None
    if any(b.GetStereo() for b in iter_bonds(mol)):
        return None

    inventory = selection_inventory(mol)
    groups = tuple(inventory.groups)
    principal = tuple(inventory.principal)
    if (not inventory.supported or len(groups) != 1 or len(principal) != 1
            or principal[0].kind != "amide" or groups[0].kind != "amide"):
        return None
    group = groups[0]
    typed = _bounded_site(mol, group)
    if typed is None:
        return None
    n_idx, o_idx, n_carbon = typed

    # This local route does not broaden the general amide suffix vocabulary:
    # every non-core heavy atom must be carbon, and all O/N atoms belong to
    # the one ordinary amide and its N-O substituent.
    if (sum(a.GetAtomicNum() == 7 for a in atoms) != 1
            or sum(a.GetAtomicNum() == 8 for a in atoms) != 2):
        return None

    carbonyl_oxygen = next((i for i in group.atoms
                            if i != group.center and i != group.ligand
                            and mol.GetAtomWithIdx(i).GetAtomicNum() == 8), None)
    if carbonyl_oxygen is None:
        return None
    # Require a simple chain or monocyclic carbon parent. The complete-parent
    # renderer and its P-44 numbering search own exact parent selection.
    carbon_atoms = [a for a in atoms if a.GetAtomicNum() == 6]
    ring_systems = {a.GetIdx() for a in carbon_atoms if a.IsInRing()}
    if ring_systems and (mol.GetRingInfo().NumRings() != 1
                         or len(ring_systems) < 3):
        return None
    if any(a.GetAtomicNum() not in {1, 6, 7, 8} for a in atoms):
        return None

    candidates = functional_parent_candidates(mol, Chem, include_single=True,
                                              principal=principal)
    heavy = _heavy(mol)
    valid = []
    for candidate in candidates:
        construction = candidate.construction
        if construction is None:
            continue
        if set(construction.atoms) | set(construction.embedded_atoms) != set(heavy):
            continue
        if set(construction.atoms) & set(construction.embedded_atoms):
            continue
        parent_atoms = set(construction.atoms) & set(heavy)
        embedded_atoms = set(construction.embedded_atoms) & set(heavy)
        if not parent_atoms or not embedded_atoms:
            continue
        o_text, o_owned = mapped_branch(mol, Chem, o_idx, n_idx)
        if not o_text or not o_owned or o_idx not in o_owned:
            continue
        n_owned = set(o_owned)
        if n_carbon is not None:
            n_text, extra_owned = mapped_branch(mol, Chem, n_carbon, n_idx)
            if not n_text or not extra_owned or n_owned & set(extra_owned):
                continue
            n_owned.update(extra_owned)
        else:
            extra_owned = set()
        if not n_owned <= embedded_atoms:
            continue
        chain_prefix_atoms = embedded_atoms - n_owned
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in chain_prefix_atoms):
            continue
        valid.append((candidate, construction, parent_atoms, n_owned,
                      set(o_owned), set(extra_owned), chain_prefix_atoms))
    if not valid:
        return None

    frontier = parent_frontier(valid, lambda row: row[0].parent_selection)
    if (not frontier or any(row[0].parent_selection is None for row in frontier)
            or not _equivalent_frontier_mappings(mol, Chem, frontier)):
        return None
    selected = frontier[0]

    (candidate, construction, parent_atoms, embedded_atoms,
     o_atoms, n_atoms, chain_prefix_atoms) = selected
    roles = [("amide-parent", frozenset(parent_atoms), construction.atom_locants)]
    roles.append(("N-O-substituent", frozenset(o_atoms), ()))
    if n_atoms:
        roles.append(("N-organyl-substituent", frozenset(n_atoms), ()))
    if chain_prefix_atoms:
        roles.append(("parent-carbon-prefixes", frozenset(chain_prefix_atoms), ()))
    if set().union(*(set(row[1]) for row in roles)) != set(heavy):
        return None
    if sum(len(row[1]) for row in roles) != len(heavy):
        return None
    return MethodDecision(
        candidate.name, "substitutive", frozenset({group.center}), heavy,
        RULE, candidate.atom_locants, component_roles=tuple(roles),
    )
