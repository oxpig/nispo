"""Mapped hydrazine parents and subordinate ylidene groups (P-68.3.1.2)."""

from dataclasses import dataclass
from functools import lru_cache

from .functional_class_queries import hydrazone_query

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


RULE = "P-41/P-44/P-45/P-68.3.1.2.1/P-68.3.1.2.2/P-16.9/P-82/P-91.3/P-93.4"


@dataclass(frozen=True, slots=True)
class HydrazoneSite:
    carbon: int
    imine_nitrogen: int  # hydrazine position 2
    terminal_nitrogen: int  # hydrazine position 1


def _order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0.0


def _sites(mol):
    found = []
    for carbon in iter_atoms(mol):
        c = carbon.GetIdx()
        if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge()
                or carbon.GetNumRadicalElectrons()):
            continue
        imine_ns = [a for a in carbon.GetNeighbors()
                    if a.GetAtomicNum() == 7 and _order(mol, c, a.GetIdx()) == 2]
        if len(imine_ns) != 1:
            continue
        n2 = imine_ns[0]
        if any(a.GetAtomicNum() not in {1, 6} for a in carbon.GetNeighbors()
               if a.GetIdx() != n2.GetIdx()):
            continue  # Hydrazides and amidrazones have distinct functional parents.
        terminal_ns = [a for a in n2.GetNeighbors()
                       if a.GetAtomicNum() == 7 and _order(mol, n2.GetIdx(), a.GetIdx()) == 1]
        if len(terminal_ns) != 1:
            continue
        n1 = terminal_ns[0]
        if (n2.GetFormalCharge() or n1.GetFormalCharge()
                or n2.GetNumRadicalElectrons() or n1.GetNumRadicalElectrons()
                or n1.IsInRing() or n2.IsInRing()
                or n2.GetTotalValence() != 3 or n1.GetTotalValence() != 3
                or _order(mol, n2.GetIdx(), n1.GetIdx()) != 1):
            continue
        # P-68 hydrazine position 2 is attached only to the ylidene carbon
        # and position 1. Position 1 may carry up to two organyl substituents.
        if any(a.GetIdx() not in {c, n1.GetIdx()} for a in n2.GetNeighbors()):
            continue
        n1_organyl = [a for a in n1.GetNeighbors() if a.GetIdx() != n2.GetIdx()
                      and a.GetAtomicNum() > 1]
        if len(n1_organyl) > 2 or any(a.GetAtomicNum() != 6 for a in n1_organyl):
            continue
        found.append(HydrazoneSite(c, n2.GetIdx(), n1.GetIdx()))
    return tuple(found)


@lru_cache(maxsize=1)
def _hydrazono_acid_query(Chem):
    return Chem.MolFromSmarts('[O;H1;+0][C;X3;!R](=[O;D1;+0])-[C;!R]=[N;X2]-[N;H1]-[#6]')


def hydrazono_acid_decision(mol, Chem):
    """Render a two-carbon acid with a source-mapped α-hydrazinylidene group."""
    from .component_mapping import mapped_general_branch
    from .extended_inventory import selection_inventory
    from .formatting import enclose_name_unit, format_prefixes, join_prefix_parent
    from .naming_result import SelectionLimitation
    from .stereochemistry import assigned_bond_stereochemistry

    matches = mol.GetSubstructMatches(_hydrazono_acid_query(Chem), maxMatches=2)
    if len(matches) != 1 or mol.GetStereoGroups() or any(
            a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetFormalCharge()
            or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery() for a in iter_atoms(mol)):
        return None
    hydroxy, acid, oxo, alpha, imine, terminal, ligand = matches[0]
    if HydrazoneSite(alpha, imine, terminal) not in _sites(mol):
        return None
    inventory = selection_inventory(mol)
    groups = inventory.principal
    if (not inventory.supported or len(groups) != 1
            or groups[0].kind != 'acid' or groups[0].center != acid):
        return None
    arms = [a for a in mol.GetAtomWithIdx(alpha).GetNeighbors() if a.GetIdx() not in {acid, imine}]
    if arms and (len(arms) != 1 or arms[0].GetAtomicNum() != 6 or not arms[0].IsInRing()
                 or _order(mol, alpha, arms[0].GetIdx()) != 1):
        # An acyclic carbon extension competes for the acid parent chain.
        return None
    occupied = {hydroxy, acid, oxo, alpha, imine, terminal}
    word, atoms = mapped_general_branch(mol, Chem, ligand, terminal)
    if not word or atoms & occupied:
        return None
    occupied.update(atoms)
    prefixes = [('2', '2-' + enclose_name_unit(word) + 'hydrazinylidene')]
    if arms:
        word, atoms = mapped_general_branch(mol, Chem, arms[0].GetIdx(), alpha)
        if not word or atoms & occupied:
            return None
        occupied.update(atoms)
        prefixes.append(('2', word))
    scope = frozenset(range(mol.GetNumAtoms()))
    if occupied != scope:
        return None
    configured = [bond for bond in iter_bonds(mol) if bond.GetStereo() or bond.HasQuery()]
    geometry = ''
    if configured:
        if (len(configured) != 1 or configured[0].HasQuery()
                or {configured[0].GetBeginAtomIdx(), configured[0].GetEndAtomIdx()} != {alpha, imine}):
            return None
        geometry = next((label for a, b, label in assigned_bond_stereochemistry(mol, Chem)
                         if {a, b} == {alpha, imine}), '')
        if geometry not in {'E', 'Z'}:
            return None
    name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), 'acetic acid')
    if geometry:
        name = f'(2{geometry})-' + name
    rule = 'P-41/P-65/P-68.3.1.2.2/P-93.4'
    return MethodDecision(name, 'substitutive', frozenset({acid}), scope, rule,
        ((acid, '1'), (alpha, '2')),
        preference_issues=(SelectionLimitation('hydrazono-acid', rule,
            'The acid parent, complete hydrazinylidene branch and original geometry are mapped.'),))


def principal_scope(mol, groups=None):
    """Return true for the bounded hydrazone or reciprocal two-arm azine."""
    sites = _sites(mol)
    if len(sites) == 1:
        centers = {sites[0].carbon}
    elif (len(sites) == 2
          and sites[0].terminal_nitrogen == sites[1].imine_nitrogen
          and sites[1].terminal_nitrogen == sites[0].imine_nitrogen
          and sites[0].carbon != sites[1].carbon):
        centers = {site.carbon for site in sites}
    else:
        return False
    if groups is not None:
        groups = getattr(groups, "principal", groups)
        if hasattr(groups, "center"):
            groups = (groups,)
        if (not groups or any(getattr(g, "kind", "") not in {"hydrazone", "imine", "azine"}
                              or getattr(g, "center", None) not in centers for g in groups)
                or {getattr(g, "center", None) for g in groups} != centers):
            return False
    return True


def _ylidene_component(mol, Chem, site):
    from .branch_rules import collect_subtree
    from .ylidene_components import carbon_ylidene

    carbon, nitrogen = site.carbon, site.imine_nitrogen
    if not mol.GetAtomWithIdx(carbon).IsInRing():
        linear = _linear_ylidene(mol, Chem, site)
        if linear is not None:
            return linear
    scope = frozenset(collect_subtree(mol, carbon, nitrogen))
    component = carbon_ylidene(mol, Chem, carbon, nitrogen, scope, terminal_contraction=True)
    if component is None:
        return None
    name, mapping = component
    return name, scope, mapping, dict(mapping)[carbon]


def _linear_ylidene(mol, Chem, site):
    """Return mapped linear carbon ylidene name and ownership, or decline."""
    from .branch_rules import parent_chain_name
    from .component_mapping import (MappedComponent, decorate_component,
                                    hydrogen_bindings)
    from .formatting import attach_suffix

    c = site.carbon
    # Traverse from the imine carbon over its saturated carbon skeleton.
    skeleton = {c}
    stack = [c]
    while stack:
        idx = stack.pop()
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing():
            return None
        for nb in atom.GetNeighbors():
            j = nb.GetIdx()
            if j in {site.imine_nitrogen, site.terminal_nitrogen} or nb.GetAtomicNum() == 1:
                continue
            if nb.GetAtomicNum() != 6 or _order(mol, idx, j) != 1:
                return None
            if j not in skeleton:
                skeleton.add(j)
                stack.append(j)
    # The bounded root is one complete unbranched path. This covers aldehyde
    # and ketone ylidene units while avoiding an unproved P-46 path choice.
    ends = [i for i in skeleton if sum(n.GetIdx() in skeleton
                                       for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 1]
    if len(skeleton) == 1:
        path = (c,)
    elif len(ends) == 2:
        def walk(start):
            result, prev, cur = [], -1, start
            while True:
                result.append(cur)
                nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                       if n.GetIdx() in skeleton and n.GetIdx() != prev]
                if not nxt:
                    break
                if len(nxt) != 1:
                    return ()
                prev, cur = cur, nxt[0]
            return tuple(result)
        options = [walk(i) for i in ends]
        path = next((p for p in options if len(p) == len(skeleton) and c in p), ())
        if not path:
            return None
        # P-14 first minimizes the ylidene locant. P-82 then compares the
        # original mapped isotope labels using the shared class-level key;
        # assigned stereo numbering is considered only after that tie.
        from .parent_annotations import isotope_seniority, stereo_numbering
        from .stereochemistry import _indexed_bond_stereochemistry

        def direction_key(candidate):
            locants = {idx: pos + 1 for pos, idx in enumerate(candidate)}
            isotope_key = isotope_seniority(mol, locants)
            stereo_map = dict(locants)
            stereo_map[site.terminal_nitrogen] = "1"
            stereo_map[site.imine_nitrogen] = "2"
            stereo_key = stereo_numbering(mol, stereo_map)
            if stereo_key is None and not _indexed_bond_stereochemistry(mol, Chem):
                stereo_key = ()
            return (candidate.index(c), isotope_key, stereo_key)

        options = (path, path[::-1])
        keys = [direction_key(candidate) for candidate in options]
        best = min(keys)
        tied = [candidate for candidate, key in zip(options, keys) if key == best]
        if len(tied) > 1:
            # Equal rule evidence must not choose between distinct spellings.
            from .formatting import attach_suffix
            spellings = [attach_suffix(parent_chain_name(mol, candidate), "ylidene",
                                       (str(candidate.index(c) + 1),))
                         for candidate in tied]
            if len(set(spellings)) > 1:
                return None
            # These are name-identical symmetry-equivalent directions; use
            # canonical graph ranks only to stabilize atom-map ownership.
            ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
            path = min(tied, key=lambda candidate: tuple(ranks[i] for i in candidate))
        else:
            path = tied[0]
    else:
        return None
    if set(path) != skeleton or any(mol.GetAtomWithIdx(i).GetIsotope()
                                    and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                    for i in skeleton):
        return None
    if any(_order(mol, a, b) != 1 for a, b in zip(path, path[1:])):
        return None
    parent = parent_chain_name(mol, path)
    if not parent:
        return None
    carbon_locant = str(path.index(c) + 1)
    ylidene = attach_suffix(parent, "ylidene", (carbon_locant,))
    # P-68's simple terminal example omits hydrazine locant 2; the carbon
    # locant internal to a ketone-derived ylidene remains part of its prefix.
    if carbon_locant == "1" and c == path[0]:
        ylidene = (parent[:-2] + "ylidene") if parent.endswith("an") else attach_suffix(parent, "ylidene")
    mapping = hydrogen_bindings(mol, {i: str(n + 1) for n, i in enumerate(path)})
    # Decorate the ylidene carbon unit without asking the component formatter
    # to own the C=N edge: that bond belongs to mapped hydrazine N-2. Its
    # original E/Z assignment is added once to the complete name below.
    isotope_graph = Chem.Mol(mol)
    edge = isotope_graph.GetBondBetweenAtoms(c, site.imine_nitrogen)
    if edge is not None:
        edge.SetStereo(Chem.BondStereo.STEREONONE)
        edge.SetBondDir(Chem.BondDir.NONE)
    component = MappedComponent(ylidene, tuple(mapping.items()), frozenset(mapping),
                                systematic_parent=True)
    from .stereochemistry import assigned_atom_stereochemistry
    original_cip = dict(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True))
    decorated_cip = dict(assigned_atom_stereochemistry(
        isotope_graph, Chem, atom_order="idx", include_pseudo=True))
    if any(original_cip.get(idx, "") != decorated_cip.get(idx, "")
           for idx in mapping):
        return None
    name = decorate_component(isotope_graph, Chem, component)
    if not name:
        return None
    return name, frozenset(mapping), tuple(mapping.items()), carbon_locant


def _ylidene_geometry(mol, Chem, site, yname, yloc):
    """Apply the original C=N E/Z assignment to the ylidene prefix (P-93)."""
    from .stereochemistry import _indexed_bond_stereochemistry

    assignments = [descriptor for a, b, descriptor in
                   _indexed_bond_stereochemistry(mol, Chem)
                   if {a, b} == {site.carbon, site.imine_nitrogen}]
    if len(assignments) > 1:
        return None
    if not assignments:
        return yname
    descriptor = assignments[0]
    # P-93.5.1.4.2.1 gives [(2E)-butan-2-ylidene] as the ylidene-prefix
    # form. If the terminal ylidene locant is omitted from its name, P-93.4.2
    # likewise omits a descriptor locant.
    stereo = f"({yloc}{descriptor})" if yloc != "1" else f"({descriptor})"
    return stereo + "-" + yname


def mapped_ylidene_prefix(mol, Chem, carbon, nitrogen, terminal_nitrogen=None, *, general=False):
    """Reuse the mapped carbon writer for a subordinate C=N prefix.

    P-68.3.1.2.2 uses the same ylidene unit when a more senior group owns
    the suffix. Annotation is transferred on the original graph; the host
    must enclose this complete unit before composing its nitrogen prefix.
    """
    site = HydrazoneSite(carbon, nitrogen,
                         nitrogen if terminal_nitrogen is None else terminal_nitrogen)
    carbonyl_derived = all(a.GetAtomicNum() in {1, 6} for a in mol.GetAtomWithIdx(carbon).GetNeighbors()
                           if a.GetIdx() != nitrogen)
    unit = (_ylidene_component(mol, Chem, site) if general and carbonyl_derived
            else _linear_ylidene(mol, Chem, site))
    if unit is None:
        return None
    name, _atoms, _mapping, locant = unit
    return _ylidene_geometry(mol, Chem, site, name, locant)


def _azine_decision(mol, Chem, sites):
    """Name a bounded neutral acyclic azine by its two hydrazine ylidene groups."""
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from .formatting import enclose_name_unit, prefix_alphabetization_key
    from .stereochemistry import assigned_atom_stereochemistry

    if len(sites) != 2 or mol.GetStereoGroups():
        return None
    left, right = sites
    if not (left.terminal_nitrogen == right.imine_nitrogen
            and right.terminal_nitrogen == left.imine_nitrogen
            and left.imine_nitrogen != right.imine_nitrogen
            and left.carbon != right.carbon):
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    if any(a.GetAtomicNum() not in {1, 6, 7} for a in iter_atoms(mol)):
        return None
    if any(a.GetIsotope() for a in iter_atoms(mol)
           if a.GetAtomicNum() == 7):
        return None
    if any(a.GetIsotope() and a.GetAtomicNum() not in {1, 6}
           for a in iter_atoms(mol)):
        return None
    nitrogen_atoms = {left.imine_nitrogen, right.imine_nitrogen}
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in nitrogen_atoms):
        return None
    # Each N belongs to the hydrazine N-N bond and exactly one C=N unit.
    if any(len([n for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1]) != 2 for i in nitrogen_atoms):
        return None
    components = []
    for site in (left, right):
        unit = _ylidene_component(mol, Chem, site)
        if unit is None:
            return None
        yname, atoms, atom_locants, yloc = unit
        if any(atoms & previous[1] for previous in components):
            return None
        yname = _ylidene_geometry(mol, Chem, site, yname, yloc)
        if not yname:
            return None
        components.append((yname, atoms, atom_locants, yloc, site))
    carbon_atoms = set().union(*(set(item[1]) for item in components))
    if carbon_atoms & nitrogen_atoms:
        return None
    if any(a.GetAtomicNum() == 6 and a.GetIdx() not in carbon_atoms
           for a in iter_atoms(mol)):
        return None
    if any(a.GetAtomicNum() == 7 and a.GetIdx() not in nitrogen_atoms
           for a in iter_atoms(mol)):
        return None
    # Symmetrical azines use the source's di(prefix) hydrazine construction;
    # unlike groups are both enclosed and cited as separate prefixes.
    same_name = components[0][0] == components[1][0]
    if (not same_name and
            prefix_alphabetization_key(components[0][0])[:3]
            == prefix_alphabetization_key(components[1][0])[:3]):
        # P-14.5 excludes isotope and stereochemical descriptors from
        # alphanumerical ordering. If those are the only distinctions between
        # unlike arms, the source gives no prefix-order tie-break.
        return None
    if same_name:
        yname = components[0][0]
        # P-68.3.1.2.3.1 directly prescribes di(propan-2-ylidene) for the
        # symmetric acetone azine example. Compound and descriptor-bearing
        # units retain the enclosure required by P-16.5.
        if "(" in yname or "[" in yname or "-" in yname:
            prefix_text = "di" + enclose_name_unit(yname)
        else:
            prefix_text = "di" + yname
    else:
        ordered = sorted(components, key=lambda item: prefix_alphabetization_key(item[0]))
        prefix_text = "".join(enclose_name_unit(item[0]) for item in ordered)
    if not prefix_text:
        return None
    name = prefix_text + "hydrazine"
    # Assign the reporting map by cited-prefix order. Equal groups are
    # symmetry-equivalent; canonical graph ranks stabilize only ownership.
    order = sorted(components, key=lambda item: prefix_alphabetization_key(item[0]))
    if order[0][0] == order[1][0]:
        ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
        order = sorted(order, key=lambda item: tuple(ranks[i] for i in sorted(item[1])))
    nmap = hydrogen_bindings(mol, {
        order[0][4].imine_nitrogen: "1",
        order[1][4].imine_nitrogen: "2",
    })
    parent_graph = Chem.Mol(mol)
    for item in components:
        edge = parent_graph.GetBondBetweenAtoms(item[4].carbon, item[4].imine_nitrogen)
        if edge is not None:
            edge.SetStereo(Chem.BondStereo.STEREONONE)
            edge.SetBondDir(Chem.BondDir.NONE)
    original_cip = dict(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True))
    decorated_cip = dict(assigned_atom_stereochemistry(
        parent_graph, Chem, atom_order="idx", include_pseudo=True))
    mapped_atoms = set(nmap)
    roles = [("hydrazine", tuple(sorted(nmap)), tuple(nmap.items()))]
    locants = list(nmap.items())
    for number, item in enumerate(order, 1):
        mapped_atoms.update(item[1])
        roles.append((f"ylidene:{number}", tuple(sorted(item[1])), item[2]))
        locants.extend(item[2])
    if any(original_cip.get(idx, "") != decorated_cip.get(idx, "")
           for idx in mapped_atoms):
        return None
    required = {a.GetIdx() for a in iter_atoms(mol)
                if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if mapped_atoms != required:
        return None
    return MethodDecision(name, "substitutive",
                          frozenset({left.carbon, right.carbon}),
                          frozenset(a.GetIdx() for a in iter_atoms(mol)
                                    if a.GetAtomicNum() > 1),
                          "P-68.3.1.2.3/P-16.5/P-82/P-91.3/P-93.4",
                          tuple(locants), (), tuple(roles))


def _principal_substituents(mol, n_atoms):
    # Halogens, ethers and aromatic heteroatoms can occur in a substituent.
    # Other characteristic groups require their own seniority comparison.
    for atom in iter_atoms(mol):
        if atom.GetIdx() in n_atoms or atom.GetAtomicNum() in {1, 6, 9, 17, 35, 53}:
            continue
        if atom.GetIsAromatic() and atom.GetAtomicNum() in {7, 8, 16}:
            continue
        if (atom.GetAtomicNum() in {8, 16} and atom.GetDegree() == 2
                and all(b.GetBondTypeAsDouble() == 1
                        and b.GetOtherAtom(atom).GetAtomicNum() == 6 for b in atom.GetBonds())):
            continue
        return False
    return True


def _hydrazine_end(mol, Chem, site, parent):
    """Map one N–N unit, its isotope hydrogens, and all terminal N substituents."""
    from .component_mapping import MappedComponent, _mapped_isotope_descriptor_plan, hydrogen_bindings, mapped_general_branch
    from .formatting import enclose_name_unit, format_prefixes

    first, second = site.imine_nitrogen, site.terminal_nitrogen
    if any(mol.GetAtomWithIdx(i).GetChiralTag() for i in (first, second)):
        return None
    mapping = hydrogen_bindings(mol, {first: '1', second: '2'})
    occupied, words = set(mapping), []
    for atom in mol.GetAtomWithIdx(second).GetNeighbors():
        root = atom.GetIdx()
        if root in mapping:
            continue
        if atom.GetAtomicNum() != 6 or _order(mol, root, second) != 1:
            return None
        word, atoms = mapped_general_branch(mol, Chem, root, second)
        if not word or atoms & occupied or site.carbon in atoms:
            return None
        occupied.update(atoms)
        words.append(word)
    prefixes = tuple(('2', word) for word in words)
    component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping), prefixes)
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, mapping)
    if plan is None:
        return None
    if parent == 'hydrazinylidene' and len(words) == 1 and not plan[0]:
        prefix = words[0] if words[0].isalpha() else enclose_name_unit(words[0])
    else:
        prefix = format_prefixes(prefixes, hierarchical_enclosures=True)
    label = '(' + ','.join(plan[0]) + ')' if plan[0] else ''
    return prefix + label + parent, frozenset(occupied), tuple(mapping.items())


def _diylidene_frames(mol, Chem, centers, link):
    from .branch_rules import parent_chain_name
    from .mapped_ring_parents import ring_systems, skeletons
    from .rooted_carbon_parents import carbon_paths

    if any(mol.GetAtomWithIdx(i).IsInRing() for i in centers):
        for ring in ring_systems(mol):
            if not centers <= ring or not ring <= link:
                continue
            for hydride, seq, locants, _, _, saturated, key, *_ in skeletons(mol, Chem, ring):
                if saturated:
                    yield hydride, dict(zip(seq, locants)), (key,), ()
        return
    for path in carbon_paths(mol, centers, link):
        for seq in (path, path[::-1]):
            orders = [_order(mol, a, b) for a, b in zip(seq, seq[1:])]
            parent = parent_chain_name(mol, seq)
            if parent:
                yield (parent, {i: str(n) for n, i in enumerate(seq, 1)},
                       (-len(seq), -sum(o > 1 for o in orders), -orders.count(2)),
                       (tuple(i for i, o in enumerate(orders, 1) if o > 1),
                        tuple(i for i, o in enumerate(orders, 1) if o == 2)))


def _bis_hydrazone_decision(mol, Chem, sites):
    """Identical hydrazine units joined by a mapped dicarbonyl-derived connector."""
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_general_branch
    from .formatting import attach_suffix, enclose_name_unit, prefix_alphabetization_key
    from .parent_annotations import isotope_numbering_key, stereo_numbering
    from .parent_selection import locant_key, locant_set

    nitrogens = {i for site in sites for i in (site.imine_nitrogen, site.terminal_nitrogen)}
    centers = frozenset(site.carbon for site in sites)
    if (len(nitrogens) != 4 or len(centers) != 2 or mol.GetStereoGroups()
            or not _principal_substituents(mol, nitrogens)
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol))):
        return None
    ends = [_hydrazine_end(mol, Chem, site, 'hydrazine') for site in sites]
    if (not all(ends) or ends[0][0] != ends[1][0] or ends[0][1] & ends[1][1]
            or any(centers & atoms for _, atoms, _ in ends)):
        return None
    # Multiplicative nomenclature requires identical repeated units, including
    # their isotopes and configurations, before numbering the connector.
    if len({Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms)) for _, atoms, _ in ends}) != 1:
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    link = scope - ends[0][1] - ends[1][1]
    options = []
    for parent, skeleton, frame_key, pi_key in _diylidene_frames(mol, Chem, centers, link):
        mapping = hydrogen_bindings(mol, skeleton)
        occupied, prefixes = set(mapping), []
        for i, loc in skeleton.items():
            for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                root = atom.GetIdx()
                if root not in link or root in occupied:
                    continue
                word, atoms = mapped_general_branch(mol, Chem, root, i)
                if word and atoms <= link and not atoms & occupied:
                    occupied.update(atoms)
                    prefixes.append((loc, word))
        if occupied != link:
            continue
        positions = sorted((skeleton[i] for i in centers), key=locant_key)
        omit = len(skeleton) == 2 and not prefixes
        parent = attach_suffix(parent, 'diylidene', () if omit else tuple(positions))
        if not parent:
            continue
        # The connector cites C=N configurations at their carbon locants.
        # The N atoms still belong to the end units, which cite their isotopes.
        imines = {site.imine_nitrogen: skeleton[site.carbon] for site in sites}
        stereo_map = mapping | imines
        component = MappedComponent(parent, tuple(stereo_map.items()), frozenset(stereo_map),
            tuple(prefixes), link - set(mapping), systematic_parent=True, local_isotope_atoms=frozenset(imines))
        connector = decorate_component(mol, Chem, component)
        if not connector:
            continue
        stereo_key = stereo_numbering(mol, stereo_map)
        isotope_key = isotope_numbering_key(mol, mapping)
        if stereo_key is None or isotope_key is None:
            continue
        citation = sorted(prefixes, key=lambda p: (prefix_alphabetization_key(p[1]), locant_key(p[0])))
        key = (frame_key, locant_set(positions), pi_key, locant_set(loc for loc, _ in prefixes),
               tuple(locant_key(loc) for loc, _ in citation), isotope_key, stereo_key)
        options.append((key, connector, tuple(mapping.items())))
    if not options:
        return None
    lowest = min(key for key, _, _ in options)
    winners = {word: mapping for key, word, mapping in options if key == lowest}
    if len(winners) != 1:
        return None
    connector, mapping = winners.popitem()
    if not connector.isalpha():
        connector = enclose_name_unit(connector)
    parent = ends[0][0]
    repeated = 'dihydrazine' if parent == 'hydrazine' else 'bis' + enclose_name_unit(parent)
    roles = tuple(('hydrazine', atoms, locants) for _, atoms, locants in ends)
    return MethodDecision("1,1'-" + connector + repeated, 'multiplicative', centers, scope,
        'P-68.3.1.2.2/P-15.3', mapping, component_roles=(*roles, ('diylidene', link, mapping)))


def hydrazinylidene_prefix(mol, Chem, root, host, atoms):
    """Account for the N-N side of a carbonyl-derived hydrazone on a senior parent."""
    from .oxime_components import imino_prefix_configuration

    other = [n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
             if n.GetIdx() != host and n.GetAtomicNum() == 7]
    if len(other) != 1:
        return '', frozenset()
    site = HydrazoneSite(host, root, other[0])
    if site not in _sites(mol):
        return '', frozenset()
    end = _hydrazine_end(mol, Chem, site, 'hydrazinylidene')
    if end is None or end[1] != atoms:
        return '', frozenset()
    return imino_prefix_configuration(mol, Chem, root, host, end[0], atoms)


def hydrazone_parent_decision(mol, Chem):
    sites = _sites(mol)
    if not sites:
        return None
    decision = hydrazone_decision(mol, Chem, sites=sites)
    if decision is not None or not mol.GetRingInfo().NumRings():
        return decision
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import records

    inventory = selection_inventory(mol)
    groups = inventory.principal
    if (not inventory.supported or not groups or
            any(g.kind not in {'acid', 'amide', 'nitrile', 'ketone', 'alcohol'} for g in groups)):
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    centers = frozenset(g.center for g in groups)
    rows = [r for r in records(mol, Chem, principal=groups, general_branches=True)
            if r.principal_centers == centers and r.hydrogen_budget_proved and r.annotation_coverage
            and r.construction is not None and r.construction.atoms | r.construction.embedded_atoms == scope]
    if not rows or len({row.name for row in rows}) != 1:
        return None
    row = rows[0]
    return MethodDecision(row.name, 'substitutive', centers, scope, 'P-41/P-68.3.1.2.2', row.atom_locants,
                          preference_issues=row.preference_issues)


def hydrazone_decision(mol, Chem, *, sites=None):
    """Construct a bounded mapped hydrazone/azine name, or return ``None``."""
    from .component_mapping import (MappedComponent, decorate_component,
                                    hydrogen_bindings)

    if sites is None:
        sites = _sites(mol)
    if len(sites) == 2:
        if sites[0].imine_nitrogen == sites[1].terminal_nitrogen:
            return _azine_decision(mol, Chem, sites)
        return _bis_hydrazone_decision(mol, Chem, sites)
    if len(sites) != 1 or mol.GetStereoGroups():
        return None
    site = sites[0]
    n_atoms = {site.imine_nitrogen, site.terminal_nitrogen}
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    if not _principal_substituents(mol, n_atoms):
        return None
    # Exactly one C=N-N centre remains after the bounded two-arm azine
    # operation above; connected higher hydrazone families remain separate.
    if sum(1 for a in iter_atoms(mol) if a.GetAtomicNum() == 7 and any(
            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetAtomicNum() == 6
            for b in a.GetBonds())) != 1:
        return None
    ylidene = _ylidene_component(mol, Chem, site)
    if ylidene is None:
        return None
    yname, yatoms, ymap, yloc = ylidene
    yname = _ylidene_geometry(mol, Chem, site, yname, yloc)
    if yname is None:
        return None
    if any(a.IsInRing() for a in (mol.GetAtomWithIdx(i) for i in n_atoms)):
        return None
    prefixes = []
    embedded = set(yatoms)
    branch_names = []
    branch_ownership = []
    for root in [a.GetIdx() for a in mol.GetAtomWithIdx(site.terminal_nitrogen).GetNeighbors()
                 if a.GetIdx() != site.imine_nitrogen and a.GetAtomicNum() > 1]:
        if mol.GetBondBetweenAtoms(site.terminal_nitrogen, root).GetBondTypeAsDouble() != 1:
            return None
        from .component_mapping import mapped_general_branch
        branch, owned = mapped_general_branch(mol, Chem, root, site.terminal_nitrogen)
        if not branch or owned & (embedded | n_atoms):
            return None
        embedded.update(owned)
        branch_names.append(branch)
        branch_ownership.append(tuple(sorted(owned)))
    if branch_names:
        from .formatting import prefix_alphabetization_key
        # Compare both possible orientations of the preselected hydrazine
        # parent. First minimize the complete locant set; only on a tie does
        # the first-cited prefix alphabetical order assign locant 1 (P-14).
        orientation_a = [("1", text) for text in branch_names] + [("2", yname)]
        orientation_b = [("2", text) for text in branch_names] + [("1", yname)]
        # Explicitly expand each N-substituent occurrence; identical prefixes
        # at one N retain their multiplicity in the locant set.
        def orientation_key(items):
            locants = tuple(sorted(int(loc) for loc, _text in items))
            first_cited = tuple(sorted(prefix_alphabetization_key(text)
                                       for loc, text in items if loc == "1"))
            return locants, first_cited
        prefixes = min((orientation_a, orientation_b), key=orientation_key)
        yloc_parent = next(loc for loc, text in prefixes if text == yname)
        branch_roles = [(f"N{loc}-organyl", atoms, ())
                        for (loc, text), atoms in zip(
                            ((loc, text) for loc, text in prefixes if text != yname),
                            branch_ownership)]
        nmap = hydrogen_bindings(mol, {
            site.terminal_nitrogen: "2" if yloc_parent == "1" else "1",
            site.imine_nitrogen: yloc_parent,
        })
    else:
        branch_roles = []
        # P-14.3.4.4: the two hydrazine sites are equivalent for a sole
        # ylidene substituent. A child carbon locant does not force a parent
        # nitrogen locant. P-16.5 retains the child's enclosing marks.
        nmap = hydrogen_bindings(mol, {
            site.terminal_nitrogen: "2", site.imine_nitrogen: "1",
        })
        # A parent isotope can distinguish the two attachment sites.
        locant = "1" if any(mol.GetAtomWithIdx(i).GetIsotope() for i in nmap) else ""
        prefixes = [(locant, yname)]
    parent_atoms = frozenset(nmap)
    parent_name = "hydrazine"
    component = MappedComponent(parent_name, tuple(nmap.items()), parent_atoms,
                                tuple(prefixes), frozenset(embedded),
                                systematic_parent=True,
                                prefix_mode='locanted' if prefixes[0][0] else 'unlocanted')
    parent_graph = Chem.Mol(mol)
    edge = parent_graph.GetBondBetweenAtoms(site.carbon, site.imine_nitrogen)
    if edge is not None:
        edge.SetStereo(Chem.BondStereo.STEREONONE)
        edge.SetBondDir(Chem.BondDir.NONE)
    from .stereochemistry import assigned_atom_stereochemistry
    original_cip = dict(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True))
    decorated_cip = dict(assigned_atom_stereochemistry(
        parent_graph, Chem, atom_order="idx", include_pseudo=True))
    if any(original_cip.get(idx, "") != decorated_cip.get(idx, "")
           for idx in parent_atoms | embedded):
        return None
    name = decorate_component(parent_graph, Chem, component)
    if not name:
        return None
    required = {a.GetIdx() for a in iter_atoms(mol)
                if a.GetAtomicNum() > 1 or a.GetIsotope()}
    covered = set(parent_atoms) | embedded
    if covered != required:
        return None
    locants = tuple(nmap.items()) + tuple(ymap)
    roles = (("hydrazine", tuple(sorted(parent_atoms)), tuple(nmap.items())),
             ("ylidene", tuple(sorted(yatoms)), tuple(ymap))) + tuple(branch_roles)
    return MethodDecision(name, "substitutive", frozenset({site.carbon}),
                          frozenset(a.GetIdx() for a in iter_atoms(mol)
                                    if a.GetAtomicNum() > 1),
                          RULE, locants, (), roles)

