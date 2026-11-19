"""Bounded mapped naming of ordinary monoazo compounds as diazenes.

The parent choice follows Blue Book P-68.3.1.3.2.1/.2. The producer admits
only a neutral, acyclic R-N=N-R' graph with two carbon-only organyl arms. It
declines higher characteristic groups, azoxy compounds, rings on the diazene
nitrogens, and branch classes not proved by this operation.
"""

from __future__ import annotations

from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .formatting import (enclose_name_unit, format_prefixes,
                         prefix_alphabetization_key)
from .naming_methods import MethodDecision


RULE = "P-68.3.1.3.2.1/P-68.3.1.3.2.2/P-14.5/P-16.5/P-82/P-93.4.2.1.3"


def _carbon_arm(mol, Chem, root, nitrogen, blocked):
    """Accept a simple alkyl tree or unsubstituted phenyl arm, with mapping."""
    text, atoms = mapped_branch(mol, Chem, root, nitrogen)
    if not text or not atoms or blocked not in {n.GetIdx() for n in
            mol.GetAtomWithIdx(root).GetNeighbors()}:
        return "", frozenset()
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
        return "", frozenset()
    ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        if len(atoms) != 6 or len(ring_atoms) != 6 or not all(
                mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return "", frozenset()
        # The P-68 source uses phenyl; additional ring substitution is outside
        # this simple phenyl construction.
        if any(n.GetIdx() not in atoms | {nitrogen} and n.GetAtomicNum() > 1
               for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()):
            return "", frozenset()
        if any(not mol.GetBondBetweenAtoms(i, j).GetIsAromatic()
               for i in atoms for j in atoms if i < j
               and mol.GetBondBetweenAtoms(i, j) is not None):
            return "", frozenset()
    else:
        # P-29 ordinary acyclic organyl arms: saturated carbon trees, plus
        # the directly published ethenyl example in P-68.3.1.3.2.2.
        unsaturated = [(i, j, mol.GetBondBetweenAtoms(i, j))
                       for i in atoms for j in atoms if i < j
                       and mol.GetBondBetweenAtoms(i, j) is not None
                       and mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1]
        if unsaturated and not (
                text == "ethenyl" and len(atoms) == 2 and len(unsaturated) == 1
                and unsaturated[0][2].GetBondTypeAsDouble() == 2):
            return "", frozenset()
        if not unsaturated:
            degrees = {i: sum(n.GetIdx() in atoms for n in
                              mol.GetAtomWithIdx(i).GetNeighbors()) for i in atoms}
            if (any(d > 2 for d in degrees.values())
                    or degrees.get(root) != (0 if len(atoms) == 1 else 1)
                    or len(atoms) > 1 and sum(d == 1 for d in degrees.values()) != 2):
                return "", frozenset()
    return text, frozenset(atoms)


def _prefix_modified(mol, atoms):
    return any(mol.GetAtomWithIdx(i).GetIsotope() or
               int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms)


def _order_arms(mol, arms):
    """Choose citation order from ordinary prefix names and P-82 modifiers.

    The final spelling field in ``prefix_alphabetization_key`` is deliberately
    excluded: isotope descriptors can alter spelling without changing the
    ordinary-prefix comparison. Core numbering is selected separately below.
    """
    from functools import cmp_to_key

    def compare(a, b):
        ka, kb = prefix_alphabetization_key(a[0])[:3], prefix_alphabetization_key(b[0])[:3]
        if ka != kb:
            return -1 if ka < kb else 1
        ma, mb = _prefix_modified(mol, a[1]), _prefix_modified(mol, b[1])
        if ma != mb:
            return -1 if ma else 1
        # Identical, otherwise indistinguishable arm names are interchangeable.
        return 0
    return sorted(arms, key=cmp_to_key(compare))


def _render_arms(arms, *, explicit_n_locants=False):
    """Render P-68 parent-substituent sequence for one or two simple arms."""
    if len(arms) != 2:
        return ""
    left, right = (x[0] for x in arms)
    if explicit_n_locants:
        prefix = format_prefixes([("1", left), ("2", right)],
                                 hierarchical_enclosures=True)
        return prefix + "diazene"
    if left == right:
        return format_prefixes([("", left), ("", right)], omit_locants=True,
                               hierarchical_enclosures=True) + "diazene"
    # P-68 examples distinguish successive substituents by enclosing units;
    # keep a first simple unit bare, while compound units are always enclosed.
    left_text = left if left.isalpha() else enclose_name_unit(left)
    right_text = enclose_name_unit(right)
    return left_text + right_text + "diazene"


def _render_numbered_arms(mol, arms, core_n_map):
    """Render citation-ordered arms with their independently assigned N locants."""
    from collections import defaultdict

    groups = defaultdict(list)
    metadata = {}
    for text, atoms, nitrogen in arms:
        groups[text].append(core_n_map[nitrogen])
        metadata[text] = (prefix_alphabetization_key(text)[:3],
                          not _prefix_modified(mol, atoms))
    # Keep P-14 ordinary-prefix citation separate from the P-82 parent
    # numbering. A label changes the prefix spelling, not its ordinary key.
    ordered_names = sorted(groups, key=lambda text: (*metadata[text], text))
    rendered = []
    for text in ordered_names:
        locants = sorted(groups[text], key=int)
        if len(locants) == 1:
            rendered.append(format_prefixes([(locants[0], text)],
                                             hierarchical_enclosures=True))
        else:
            rendered.append(format_prefixes([(locant, text) for locant in locants],
                                             hierarchical_enclosures=True))
    return "-".join(rendered) + "diazene"


def _core_isotope_numbering_key(mol, ordered):
    """P-82.5.2 key for all core and isotopically modified arm entities.

    Shared annotation helpers supply original-graph isotope facts. First
    compare the complete locant series for modified parent atoms and groups;
    if tied, higher atomic number and then higher mass number gets the lower
    locant. A labeled arm contributes its attachment locant as a modified
    group, while its isotope atom supplies the species comparison.
    """
    from .parent_annotations import isotope_numbering_key
    mapping = {row[2]: str(pos) for pos, row in enumerate(ordered, 1)}
    return isotope_numbering_key(mol, mapping, include_substituent_groups=True)


def diazene_decision(mol, Chem):
    """Return a complete mapped decision for the proved monoazo scope."""
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() not in {1, 6, 7} for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    n_bonds = [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2
               and b.GetBeginAtom().GetAtomicNum() == 7
               and b.GetEndAtom().GetAtomicNum() == 7]
    if len(nitrogens) != 2 or len(n_bonds) != 1:
        return None
    bond = n_bonds[0]
    n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in (n1, n2)):
        return None
    if any(a.GetAtomicNum() == 6 and a.GetIsAromatic() and not a.IsInRing()
           for a in mol.GetAtoms()):
        return None
    # Exactly one carbon attachment at each diazene N; no N-H arms are
    # admitted by this whole-azo-parent rule.
    roots = {}
    for n in (n1, n2):
        others = [x.GetIdx() for x in mol.GetAtomWithIdx(n).GetNeighbors()
                  if x.GetIdx() != (n2 if n == n1 else n1)
                  and x.GetAtomicNum() > 1]
        if len(others) != 1 or mol.GetAtomWithIdx(others[0]).GetAtomicNum() != 6:
            return None
        if mol.GetBondBetweenAtoms(n, others[0]).GetBondTypeAsDouble() != 1:
            return None
        roots[n] = others[0]
    # Avoid this producer overriding ordinary characteristic-group parents.
    # Any O, S, P, halogen, or other non-C/N element is outside admitted arms.
    if any(a.GetAtomicNum() not in {1, 6, 7} for a in mol.GetAtoms()):
        return None
    arm_rows = []
    for n in (n1, n2):
        text, atoms = _carbon_arm(mol, Chem, roots[n], n, n)
        if not text:
            return None
        arm_rows.append((text, atoms, n))
    if arm_rows[0][1] & arm_rows[1][1]:
        return None
    if any(nb.GetAtomicNum() == 1 and nb.GetIsotope()
           for i in (n1, n2) for nb in mol.GetAtomWithIdx(i).GetNeighbors()):
        return None
    ordered = _order_arms(mol, arm_rows)
    # The unsubstituted name fixes citation order first. If the two arms are
    # chemically equivalent, P-82.5.2 chooses the direction with the lowest
    # locant series for modified core atoms; for mixed N nuclides the heavier
    # mass gets the first locant if the modified-locant series ties.
    chemically_equivalent_arms = (
        prefix_alphabetization_key(ordered[0][0])[:3] ==
        prefix_alphabetization_key(ordered[1][0])[:3]
    )
    numbering = ordered
    if chemically_equivalent_arms:
        reverse = list(reversed(ordered))
        first_key = _core_isotope_numbering_key(mol, ordered)
        reverse_key = _core_isotope_numbering_key(mol, reverse)
        if first_key is None or reverse_key is None:
            return None
        if reverse_key < first_key:
            numbering = reverse
    core_n_map = {row[2]: str(i + 1) for i, row in enumerate(numbering)}
    core = {n1, n2}
    component = MappedComponent(
        "diazene", tuple(core_n_map.items()), frozenset(core),
        embedded_atoms=frozenset(set().union(*(set(row[1]) for row in ordered))),
        nitrogen_site_locants=True,
    )
    # Use the independent P-68 construction above; the component decorator is
    # responsible only for mapped N-isotope formatting, not arm ordering.
    copy = Chem.Mol(mol)
    copied_bond = copy.GetBondBetweenAtoms(n1, n2)
    copied_bond.SetStereo(Chem.BondStereo.STEREONONE)
    copied_bond.SetBondDir(Chem.BondDir.NONE)
    core_text = decorate_component(copy, Chem, component)
    if not core_text:
        return None
    nitrogen_masses = [mol.GetAtomWithIdx(i).GetIsotope() for i in (n1, n2)]
    explicit_n_locants = any(nitrogen_masses) and len(set(nitrogen_masses)) > 1
    if explicit_n_locants:
        base = _render_numbered_arms(mol, ordered, core_n_map)
    else:
        base = _render_arms(ordered, explicit_n_locants=False)
    if not base:
        return None
    # The diazene isotope descriptor belongs immediately before the parent
    # hydride term, after its substituent prefixes (P-82.2.1).
    isotope = core_text[:-len("diazene")] if core_text.endswith("diazene") else ""
    name = base[:-len("diazene")] + isotope + "diazene"
    if bond.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
        descriptor = "E" if bond.GetStereo() == Chem.BondStereo.STEREOE else "Z"
        descriptor = ("1" if explicit_n_locants else "") + descriptor
        name = f"({descriptor})-" + name
    elif int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE:
        return None
    mapped = set(core)
    roles = [("diazene", frozenset(core), tuple(core_n_map.items()))]
    for pos, (text, atoms, n) in enumerate(ordered, 1):
        if mapped & set(atoms):
            return None
        mapped.update(atoms)
        roles.append((f"organyl-{pos}", frozenset(atoms), ()))
    required = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if mapped != required:
        return None
    if any(a.GetAtomicNum() == 7 and a.GetIdx() not in core for a in mol.GetAtoms()):
        return None
    return MethodDecision(name, "substitutive", frozenset(core),
                          frozenset(required), RULE,
                          tuple(core_n_map.items()), (), tuple(roles))


