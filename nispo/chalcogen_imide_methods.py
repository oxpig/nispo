"""Mapped ordinary S/Se/Te imide parents from Blue Book P-68.4.3.3–.8."""

from collections import Counter
from dataclasses import dataclass

from .component_mapping import mapped_branch
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = "P-14.1.3/P-14.5/P-16.3/P-16.5.1.3.1/P-68.4.3.3/P-68.4.3.4/P-68.4.3.5/P-68.4.3.6/P-68.4.3.7/P-68.4.3.8"
_ELEMENTS = {16: ("S", "sulf", "sulfan", "sulfane"),
             34: ("Se", "selen", "selan", "selane"),
             52: ("Te", "tellur", "tellan", "tellane")}


@dataclass(frozen=True, slots=True)
class Organyl:
    root: int
    atoms: frozenset[int]
    name: str
    owner: int


@dataclass(frozen=True, slots=True)
class ChalcogenImide:
    center: int
    element: int
    lambda_number: int
    imino_nitrogens: tuple[int, ...]
    oxo_oxygens: tuple[int, ...]
    center_organyls: tuple[Organyl, ...]
    nitrogen_organyls: tuple[Organyl, ...]
    family: str


def _is_single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1.0


def _plain_organyl(mol, Chem, root, owner):
    """Return one unbranched alkyl or unsubstituted phenyl component."""
    if not _is_single(mol, root, owner):
        return None
    atoms, pending = {root}, [root]
    while pending:
        idx = pending.pop()
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6:
            return None
        for neighbor in atom.GetNeighbors():
            ni = neighbor.GetIdx()
            if ni == owner or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            bond = mol.GetBondBetweenAtoms(idx, ni)
            if ni not in atoms:
                atoms.add(ni)
                pending.append(ni)
    owned = frozenset(atoms)
    ring = {i for i in owned if mol.GetAtomWithIdx(i).IsInRing()}
    if ring:
        if (len(owned) != 6 or ring != set(owned)
                or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in owned)):
            return None
        edge_count = sum(1 for bond in mol.GetBonds()
                         if bond.GetBeginAtomIdx() in owned and bond.GetEndAtomIdx() in owned)
        if edge_count != 6:
            return None
    else:
        if len(owned) > 10:
            return None
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in owned):
            return None
        if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1.0
               for a in owned for b in (n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
               if b in owned):
            return None
        degree = {i: sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                         if n.GetAtomicNum() == 6 and n.GetIdx() in owned) for i in owned}
        if max(degree.values(), default=0) > 2 or sum(d == 1 for d in degree.values()) != (0 if len(owned) == 1 else 2):
            return None
        if degree.get(root, 0) > 1:
            return None
    # The shared branch renderer supplies the component spelling and verifies
    # that the original cut remains a single-bond alkyl/phenyl attachment.
    name, rendered_atoms = mapped_branch(mol, Chem, root, owner)
    if not name or frozenset(rendered_atoms) != owned:
        return None
    return Organyl(root, owned, name, owner)


def recognize_chalcogen_imide(mol, Chem):
    """Recognize one exact, neutral, ordinary P-68.4.3.3–.8 core graph."""
    if (mol is None or mol.GetNumAtoms() > 128 or mol.GetStereoGroups()
            or any(atom.GetIsotope() or atom.GetFormalCharge()
                   or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                   for atom in mol.GetAtoms())
            or any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
            or any(int(bond.GetStereo()) for bond in mol.GetBonds())):
        return None
    # The carbon-free parent hydrides are preselected/inorganic components;
    # this producer constructs organic names only (P-10/P-12).
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in _ELEMENTS]
    if len(centers) != 1:
        return None
    center = centers[0]
    eidx, element = center.GetIdx(), center.GetAtomicNum()
    if center.IsInRing():
        return None
    imino, oxos, center_roots = [], [], []
    for neighbor in center.GetNeighbors():
        idx = neighbor.GetIdx()
        order = mol.GetBondBetweenAtoms(eidx, idx).GetBondTypeAsDouble()
        if neighbor.GetAtomicNum() == 7 and order == 2.0:
            imino.append(idx)
        elif (neighbor.GetAtomicNum() == 8 and order == 2.0
              and neighbor.GetDegree() == 1):
            oxos.append(idx)
        elif neighbor.GetAtomicNum() == 6 and order == 1.0:
            center_roots.append(idx)
        else:
            return None
    imino, oxos = tuple(sorted(imino)), tuple(sorted(oxos))
    if not 1 <= len(imino) <= 3 or len(oxos) > 2 or len(center_roots) > 2:
        return None
    center_organyls = []
    for root in center_roots:
        group = _plain_organyl(mol, Chem, root, eidx)
        if group is None:
            return None
        center_organyls.append(group)
    nitrogen_organyls = []
    for nidx in imino:
        nitrogen = mol.GetAtomWithIdx(nidx)
        if nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3:
            return None
        neighbors = [a for a in nitrogen.GetNeighbors() if a.GetIdx() != eidx]
        if len(neighbors) == 0:
            if nitrogen.GetTotalNumHs() != 1:
                return None
            continue
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() != 6:
            return None
        group = _plain_organyl(mol, Chem, neighbors[0].GetIdx(), nidx)
        if group is None or nitrogen.GetTotalNumHs() != 0:
            return None
        nitrogen_organyls.append(group)
    if any(a.GetAtomicNum() not in {1, 6, 7, 8, element} for a in mol.GetAtoms()):
        return None
    groups = center_organyls + nitrogen_organyls
    all_heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    owned = {eidx, *imino, *oxos}
    for group in groups:
        if owned & set(group.atoms):
            return None
        owned.update(group.atoms)
    if owned != all_heavy:
        return None
    if center.GetTotalValence() not in {4, 6}:
        return None
    lam = int(center.GetTotalValence())
    family = _family(len(imino), len(oxos), lam, center_organyls, nitrogen_organyls)
    if family is None:
        return None
    return ChalcogenImide(eidx, element, lam, imino, oxos,
                          tuple(center_organyls), tuple(nitrogen_organyls), family)


def _family(n_count, o_count, lam, center_groups, nitrogen_groups):
    n_names = [g.name for g in nitrogen_groups]
    if n_count == 1 and o_count == 0 and lam == 4:
        if len(center_groups) not in {0, 2} or len(n_names) > 1:
            return None
        return "sulfimide"
    if n_count == 1 and o_count == 1 and lam == 4:
        if center_groups or len(n_names) != 1:
            return None
        return "sulfinylamine"
    if n_count == 1 and o_count == 2 and lam == 6:
        if center_groups or len(n_names) != 1:
            return None
        return "sulfonylamine"
    if n_count == 1 and o_count == 1 and lam == 6:
        if len(center_groups) != 2 or len(n_names) != 1:
            return None
        return "sulfoximide"
    if n_count == 2 and o_count == 0 and lam == 6:
        if len(center_groups) != 2 or len(n_names) != 0:
            return None
        return "sulfonediimine"
    if n_count == 2 and o_count == 0 and lam == 4:
        if center_groups or len(n_names) not in {0, 2}:
            return None
        return "sulfur_diimide"
    if n_count == 3 and o_count == 0 and lam == 6:
        if center_groups or len(n_names) not in {0, 3}:
            return None
        return "sulfur_triimide"
    return None


def _multiplicative_component(names):
    """Cite mapped simple groups on a mononuclear parent under P-16.5.1.3.1."""
    if not names:
        return ""
    sequence = mononuclear_prefix_sequence(PrefixUnit(name, "simple") for name in names)
    return sequence.text if sequence else None


def _parent_text(site):
    _, _, stem, stem_e = _ELEMENTS[site.element]
    family = site.family
    if family == "sulfimide":
        tail = stem + "imine"
    elif family == "sulfinylamine" or family == "sulfoximide":
        tail = stem + "one"
    elif family == "sulfonylamine":
        tail = stem_e + "dione"
    elif family in {"sulfonediimine", "sulfur_diimide"}:
        tail = stem_e + "diimine"
    else:
        tail = stem_e + "triimine"
    return f"lambda{site.lambda_number}-{tail}"


def _n_pair_prefix(names):
    if not names:
        return ""
    if len(names) != 2:
        return None
    return _multiplicative_component(names)


def _triimine_prefix(names):
    if not names:
        return ""
    if len(names) != 3:
        return None
    return _multiplicative_component(names)


def _ordered_parent_prefixes(units):
    """Render typed parent substituent units after global P-14.5 ordering.

    Each unit is ``(unmultiplied component, count, nature)``. Compound imino
    groups retain their required enclosure even when cited first; later simple
    groups follow P-16.5.1.3.1 with the multiplier outside parentheses.
    """
    sequence = mononuclear_prefix_sequence(
        PrefixUnit(component, nature, count) for component, count, nature in units
    )
    return sequence.text if sequence else ""


def _render_name(site):
    parent = _parent_text(site)
    c_names = [group.name for group in site.center_organyls]
    n_names = [group.name for group in site.nitrogen_organyls]
    if site.family == "sulfimide":
        prefixes = []
        if c_names:
            element = _ELEMENTS[site.element][0]
            if len(set(c_names)) == 1:
                group_text = _multiplicative_component(c_names)
                if group_text is None:
                    return ""
                prefixes.append((prefix_alphabetization_key(c_names[0]),
                                 f"{element},{element}-{group_text}"))
            else:
                for name in sorted(c_names, key=prefix_alphabetization_key):
                    prefixes.append((prefix_alphabetization_key(name),
                                     f"{element}-{name}"))
        if n_names:
            prefixes.append((prefix_alphabetization_key(n_names[0]), "N-" + n_names[0]))
        return "-".join([text for _, text in sorted(prefixes)] + [parent])
    if site.family in {"sulfinylamine", "sulfonylamine", "sulfoximide"}:
        units = [(name, count, "simple")
                 for name, count in Counter(c_names).items()]
        units.extend((name + "imino", count, "compound")
                     for name, count in Counter(n_names).items())
        prefixes = _ordered_parent_prefixes(units)
        if units and not prefixes:
            return ""
        return prefixes + ("-" if prefixes else "") + parent
    if site.family == "sulfonediimine":
        c_prefix = _multiplicative_component(c_names)
        if c_prefix is None:
            return ""
        return (c_prefix + "-" if c_prefix else "") + parent
    if site.family == "sulfur_diimide":
        n_prefix = _n_pair_prefix(n_names)
        if n_prefix is None:
            return ""
        return (n_prefix + "-" if n_prefix else "") + parent
    n_prefix = _triimine_prefix(n_names)
    if n_prefix is None:
        return ""
    return (n_prefix + "-" if n_prefix else "") + parent


def chalcogen_imide_decision(mol, Chem):
    """Return a complete mapped substitutive λ-parent construction, or None."""
    site = recognize_chalcogen_imide(mol, Chem)
    if site is None:
        return None
    name = _render_name(site)
    if not name:
        return None
    center_locants = ((site.center, _ELEMENTS[site.element][0]),) if site.family == "sulfimide" and site.center_organyls else ()
    roles = [("chalcogen_center", frozenset({site.center}), center_locants)]
    if site.oxo_oxygens:
        roles.append(("oxo_oxygen", frozenset(site.oxo_oxygens), ()))
    imino_locants = ((site.imino_nitrogens[0], "N"),) if site.family == "sulfimide" and site.nitrogen_organyls else ()
    roles.append(("imino_nitrogen", frozenset(site.imino_nitrogens), imino_locants))
    for i, group in enumerate(site.center_organyls, 1):
        roles.append((f"chalcogen_organyl:{i}", group.atoms, ()))
    for i, group in enumerate(site.nitrogen_organyls, 1):
        roles.append((f"imino_organyl:{i}", group.atoms, ()))
    atom_sets = [set(atoms) for _, atoms, _ in roles]
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (any(atom_sets[i] & atom_sets[j] for i in range(len(atom_sets))
            for j in range(i + 1, len(atom_sets)))
            or frozenset().union(*(frozenset(x) for x in atom_sets)) != heavy):
        return None
    atom_locants = center_locants + imino_locants
    return MethodDecision(name=name, method="substitutive",
        principal_centers=frozenset({site.center}), atoms=heavy, rule=RULE,
        atom_locants=atom_locants, component_roles=tuple(roles))
