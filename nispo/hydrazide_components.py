"""Bounded atom-mapped carbon and sulfur oxoacid hydrazides (P-66.3).

Recognition is deliberately separate from naming: it proves the local
neutral C(=O)-N-N or S(=O)(=O)-N-N / S(=O)-N-N topology and retains
the input atom identities. The proposal
helper reuses shared acid-capped parent selection, then translates the typed
construction to the original graph for N/N′ substituents, isotope, and CIP
rendering. Mono-site and grouped-site candidates are provided; global method
and seniority decisions remain with the caller.
"""

from dataclasses import dataclass, replace


RULES = ("P-41-class-12", "P-44.1.1/P-45.2", "P-66.3.0", "P-66.3.1.1",
         "P-66.3.1.2.1", "P-66.3.3.1", "P-16.9")


def thiosemicarbazone_site(sulfur):
    """Map the drawn S=C(N)-NH-N=C derivative of a thiocarboxamide."""
    if sulfur.GetDegree() != 1 or sulfur.GetFormalCharge() or sulfur.GetIsotope():
        return None
    bond = sulfur.GetBonds()[0]
    center = bond.GetOtherAtom(sulfur)
    if (bond.GetBondTypeAsDouble() != 2 or center.GetAtomicNum() != 6
            or center.IsInRing() or center.GetDegree() != 3):
        return None
    first_nitrogens = [b.GetOtherAtom(center) for b in center.GetBonds()
                       if b.GetBondTypeAsDouble() == 1 and b.GetOtherAtom(center).GetAtomicNum() == 7]
    if len(first_nitrogens) != 2:
        return None
    mol = sulfur.GetOwningMol()
    for amino, hydrazino in (first_nitrogens, first_nitrogens[::-1]):
        if hydrazino.IsInRing() or hydrazino.GetDegree() != 2 or hydrazino.GetTotalNumHs() != 1:
            continue
        imino = next(a for a in hydrazino.GetNeighbors() if a.GetIdx() != center.GetIdx())
        if (imino.GetAtomicNum() != 7 or imino.IsInRing() or imino.GetDegree() != 2
                or _bond_order(mol, hydrazino.GetIdx(), imino.GetIdx()) != 1):
            continue
        other = next(b for b in imino.GetBonds() if b.GetOtherAtomIdx(imino.GetIdx()) != hydrazino.GetIdx())
        ylidene = other.GetOtherAtom(imino)
        if other.GetBondTypeAsDouble() != 2 or ylidene.GetAtomicNum() != 6:
            continue
        atoms = (center, sulfur, amino, hydrazino, imino)
        if any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
               or a.GetChiralTag() for a in atoms):
            continue
        return tuple(a.GetIdx() for a in atoms), ylidene.GetIdx()
    return None


def thiosemicarbazone_decision(mol, Chem, inventory):
    """Use the P-68.3.1.2.5 parent and retain the separate ylidene E/Z."""
    if not inventory.supported or len(inventory.principal) != 1:
        return None
    group = inventory.principal[0]
    if group.kind != 'thiosemicarbazone' or mol.GetStereoGroups():
        return None
    from .branch_rules import collect_subtree
    from .formatting import format_prefixes
    from .mapped_ring_parents import records
    from .naming_methods import MethodDecision
    from .nitrogen_acid_derivatives import _local_configuration, _ligands
    from .naming_result import SelectionLimitation

    center, sulfur, amino, _hydrazino, imino = group.atoms
    site = thiosemicarbazone_site(mol.GetAtomWithIdx(sulfur))
    if site is None or site[0] != group.atoms:
        return None
    _, root = site
    owned, roles = set(group.atoms), []
    cyclic_amino = mol.GetAtomWithIdx(amino).IsInRing()
    if cyclic_amino:
        amino_atoms = frozenset(collect_subtree(mol, amino, center))
        if owned & amino_atoms != {amino}:
            return None
        owned.update(amino_atoms)
        roles.append(('cyclic-N-substituent', amino_atoms, ()))
    else:
        ligands = _ligands(mol, Chem, amino, center, owned, roles)
        if ligands is None:
            return None
    scope = frozenset(collect_subtree(mol, root, imino))
    if (not mol.GetAtomWithIdx(root).IsInRing() or owned & scope
            or owned | scope != set(range(mol.GetNumAtoms()))
            or any(mol.GetAtomWithIdx(i).GetChiralTag() or mol.GetAtomWithIdx(i).GetIsotope()
                   for i in scope)):
        return None
    stereo = _local_configuration(mol, Chem, root, imino)
    if stereo is None:
        return None
    graph = Chem.Mol(mol)
    graph.GetBondBetweenAtoms(root, imino).SetStereo(Chem.BondStereo.STEREONONE)
    rows = records(graph, Chem, scope=scope, attachment=root, general_branches=True)
    if len(rows) == 1 and rows[0].annotation_coverage and rows[0].hydrogen_budget_proved:
        text, locants = rows[0].name, rows[0].atom_locants
    else:
        from .polycyclic_rules import mapped_fused_polycycle_attachment

        ylidene = mapped_fused_polycycle_attachment(graph, Chem, root, imino, scope)
        if ylidene is None:
            return None
        text, locants = ylidene
    ylidene = (f'({stereo})-' if stereo else '') + text
    if cyclic_amino:
        name = _cyclic_thiohydrazide_name(mol, Chem, amino, center, amino_atoms, ylidene)
    else:
        prefixes = [('N', text) for text in ligands] + [('2', ylidene)]
        name = format_prefixes(prefixes, hierarchical_enclosures=True) + 'hydrazine-1-carbothioamide'
    if not name:
        return None
    roles += [('ylidene', scope, locants),
              ('carbothioamide', frozenset(group.atoms) - ({amino} if cyclic_amino else set()), ())]
    rule = 'P-68.3.1.2.5/P-66.3.4'
    return MethodDecision(name, 'substitutive', frozenset({center}), frozenset(owned | scope),
                          rule, component_roles=tuple(roles), preference_issues=(
                              SelectionLimitation('thioamide-parent-comparison', rule,
                                  'The original thioamide and ylidene are mapped. Parent comparison remains open.'),))


def _cyclic_thiohydrazide_name(mol, Chem, amino, center, scope, ylidene):
    """Use a numbered cyclic N as the host of the thiocarbonyl suffix."""
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .formatting import attach_suffix, prefix_alphabetization_key
    from .heteropolycyclic_parents import skeletons
    from .mapped_ring_parents import _core_graph, ring_systems
    from .parent_selection import locant_key, locant_set
    from .parent_annotations import stereo_numbering

    core = next((ring for ring in ring_systems(mol) if amino in ring), frozenset())
    if not core <= scope or center in scope or not 3 <= len(core) <= 13:
        return ''
    branches, occupied = {}, set(core)
    for host in core:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in core or (host, root) == (amino, center):
                continue
            text, atoms = mapped_general_branch(mol, Chem, root, host)
            if not text or atoms & occupied or not atoms <= scope:
                return ''
            branches[host, root] = text
            occupied.update(atoms)
    if occupied != scope:
        return ''
    options = []
    capped = _core_graph(mol, Chem, core)
    if capped is None:
        return ''
    graph, original = capped
    for parent, seq, locants, _numbers, _edges, saturated, key, *_ in skeletons(graph):
        if not saturated:
            continue
        mapping = dict(zip((original[i] for i in seq), locants, strict=True))
        locant = mapping[amino]
        rows = [(mapping[host], text) for (host, _root), text in branches.items()]
        rank = (key, locant_key(locant), locant_set(loc for loc, _text in rows),
                locant_set(loc for loc, _text in sorted(rows,
                    key=lambda row: prefix_alphabetization_key(row[1]))))
        stereo = stereo_numbering(mol, mapping)
        if stereo is None:
            continue
        rows.append(("N'", ylidene))
        name = decorate_component(mol, Chem, MappedComponent(
            attach_suffix(parent, 'carbothiohydrazide', (locant,)),
            tuple(mapping.items()), core, tuple(rows), scope - core,
            systematic_parent=True, nitrogen_site_locants=True))
        if name:
            options.append(((*rank, stereo), name))
    if not options:
        return ''
    best = min(rank for rank, _name in options)
    names = {name for rank, name in options if rank == best}
    return names.pop() if len(names) == 1 else ''


@dataclass(frozen=True, slots=True)
class HydrazideSite:
    center: int
    carbonyl_oxygen: int
    acyl_nitrogen: int
    terminal_nitrogen: int
    atoms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HydrazideProposal:
    """Mapped acid-derived suffix candidate; not yet a cross-route decision."""
    name: str
    site: HydrazideSite
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    n_locants: tuple[tuple[int, str], ...]
    parent_selection: object = None
    prefixes: tuple[tuple[str, str], ...] = ()
    embedded_atoms: frozenset[int] = frozenset()
    annotation_coverage: bool = True
    construction: object = None
    route: str = "mapped-hydrazide"
    principal_kind: str = "hydrazide"
    rule_basis: tuple[str, ...] = RULES

    @property
    def principal_centers(self):
        return frozenset({self.site.center})

    def as_parent_proposal(self):
        """Adapt this mapped candidate to the shared cross-route proposal API."""
        from .cross_route_selection import ParentProposal
        return ParentProposal(
            self.name, self.route, self.principal_kind, self.principal_centers,
            self.parent_atoms, self.atom_locants, self.parent_selection,
            self.annotation_coverage, construction=self.construction)


@dataclass(frozen=True, slots=True)
class PolyHydrazideProposal:
    """One shared mapped parent construction for multiple hydrazide sites."""
    name: str
    sites: tuple[HydrazideSite, ...]
    site_locants: tuple[tuple[int, str], ...]
    parent_atoms: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    n_locants: tuple[tuple[int, str], ...]
    parent_selection: object = None
    prefixes: tuple[tuple[str, str], ...] = ()
    embedded_atoms: frozenset[int] = frozenset()
    annotation_coverage: bool = True
    construction: object = None
    route: str = "mapped-hydrazide"
    principal_kind: str = "hydrazide"
    rule_basis: tuple[str, ...] = RULES + ("P-66.3.3.2",)

    @property
    def principal_centers(self):
        return frozenset(site.center for site in self.sites)

    def as_parent_proposal(self):
        """Adapt the grouped candidate to the existing shared proposal API."""
        from .cross_route_selection import ParentProposal
        return ParentProposal(
            self.name, self.route, self.principal_kind, self.principal_centers,
            self.parent_atoms, self.atom_locants, self.parent_selection,
            self.annotation_coverage, construction=self.construction)


def _bond_order(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond.GetBondTypeAsDouble() if bond else 0


def _ordinary_site(mol, center, *, annular_terminal=False):
    carbon = mol.GetAtomWithIdx(center)
    if carbon.GetAtomicNum() not in {6, 16} or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons():
        return None
    oxygens = [a.GetIdx() for a in carbon.GetNeighbors()
               if a.GetAtomicNum() == 8 and _bond_order(mol, center, a.GetIdx()) == 2]
    nitrogens = [a.GetIdx() for a in carbon.GetNeighbors()
                 if a.GetAtomicNum() == 7 and _bond_order(mol, center, a.GetIdx()) == 1]
    if len(nitrogens) != 1 or (len(oxygens) != 1 if carbon.GetAtomicNum() == 6
                              else len(oxygens) not in {1, 2}):
        return None
    if carbon.GetAtomicNum() == 16:
        organyl = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 6
                   and _bond_order(mol, center, a.GetIdx()) == 1]
        if (carbon.IsInRing() or len(organyl) != 1
                or carbon.GetDegree() != len(oxygens) + 2):
            return None
    n1 = nitrogens[0]
    if _bond_order(mol, center, oxygens[0]) != 2:
        return None
    n2s = [a.GetIdx() for a in mol.GetAtomWithIdx(n1).GetNeighbors()
           if a.GetIdx() != center and a.GetAtomicNum() == 7
           and _bond_order(mol, n1, a.GetIdx()) == 1]
    if len(n2s) != 1:
        return None
    n2 = n2s[0]
    if any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53} or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    # P-66.3.3 permits independent substitution on N and N′. The acyl-bound
    # nitrogen has one substituent slot beyond C(O)-N-N; the terminal nitrogen
    # has two, including the explicitly illustrated N′,N′-dialkyl pattern.
    for index, other, max_organyl in ((n1, {center, n2}, 1), (n2, {n1}, 2)):
        atom = mol.GetAtomWithIdx(index)
        heavy = [a.GetIdx() for a in atom.GetNeighbors() if a.GetAtomicNum() > 1]
        if (atom.GetIsAromatic() or atom.GetTotalValence() != 3
                or atom.IsInRing() and (index != n2 or not annular_terminal)):
            return None
        if len([i for i in heavy if i not in other]) > max_organyl:
            return None
        for i in heavy:
            if i in other:
                continue
            branch = mol.GetAtomWithIdx(i)
            if branch.GetAtomicNum() != 6 or _bond_order(mol, index, i) != 1:
                return None
            if any(a.GetAtomicNum() == 7 and a.GetIdx() not in {n1, n2}
                   for a in branch.GetNeighbors()):
                return None
    if _bond_order(mol, n1, n2) != 1:
        return None
    # Carbonyl N must have exactly one acyl neighbor; terminal N cannot be
    # attached to another carbonyl or to an azide/hydrazone extension.
    for nidx in (n1, n2):
        for atom in mol.GetAtomWithIdx(nidx).GetNeighbors():
            if atom.GetAtomicNum() != 6 or atom.GetIdx() in {center}:
                continue
            if any(a.GetAtomicNum() == 8 and _bond_order(mol, atom.GetIdx(), a.GetIdx()) == 2
                   for a in atom.GetNeighbors()):
                return None
    return HydrazideSite(center, oxygens[0], n1, n2,
                         (center, *oxygens, n1, n2))


def recognize_hydrazides(mol, *, annular_terminal=False):
    """Return each proven ordinary neutral acyl hydrazide site in graph order."""
    found = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {6, 16}:
            continue
        site = _ordinary_site(mol, atom.GetIdx(), annular_terminal=annular_terminal)
        if site is not None:
            found.append(site)
    return tuple(found)


def _potential_hydrazide_centers(mol):
    """Find every carbon/sulfur oxoacid N-N candidate, including deferred forms."""
    centers = set()
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() not in {6, 16}:
            continue
        has_oxo = any(neighbor.GetAtomicNum() == 8
                      and _bond_order(mol, carbon.GetIdx(), neighbor.GetIdx()) == 2
                      for neighbor in carbon.GetNeighbors())
        if not has_oxo:
            continue
        for nitrogen in carbon.GetNeighbors():
            if (nitrogen.GetAtomicNum() == 7
                    and _bond_order(mol, carbon.GetIdx(), nitrogen.GetIdx()) == 1
                    and any(neighbor.GetAtomicNum() == 7
                            and _bond_order(mol, nitrogen.GetIdx(), neighbor.GetIdx()) == 1
                            for neighbor in nitrogen.GetNeighbors())):
                centers.add(carbon.GetIdx())
                break
    return frozenset(centers)


def unsubstituted_hydrazide_prefix(mol, root, blocked, atoms):
    """P-66.3.2.1: a complete unsubstituted acyl hydrazide substituent.

    This is the acyl-attached characteristic group, not the reverse
    N-attached hydrazido construction of P-66.3.2.3. Other N substitution and
    characteristic-atom annotations keep their unresolved compatibility route.
    """
    center = mol.GetAtomWithIdx(root)
    if (center.GetAtomicNum() not in {6, 16}
            or _bond_order(mol, root, blocked) != 1):
        return None
    site = _ordinary_site(mol, root)
    if site is None:
        return None
    if set(site.atoms) != set(atoms):
        return '', frozenset()
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           or int(mol.GetAtomWithIdx(i).GetChiralTag())
           or any(a.GetAtomicNum() == 1 and a.GetIsotope()
                  for a in mol.GetAtomWithIdx(i).GetNeighbors()) for i in site.atoms):
        return '', frozenset()
    if any(a.GetAtomicNum() > 1 and a.GetIdx() not in set(site.atoms) | {blocked}
           for i in site.atoms for a in mol.GetAtomWithIdx(i).GetNeighbors()):
        return '', frozenset()
    if center.GetAtomicNum() == 6:
        return 'hydrazinecarbonyl', atoms
    return ('hydrazinesulfonyl' if len(site.atoms) == 5 else 'hydrazinesulfinyl'), atoms


def _replace_acid_ending(name):
    """Apply P-66.3's bounded retained/systematic acid-to-hydrazide forms."""
    retained = {
        "formic acid": "formohydrazide",
        "acetic acid": "acetohydrazide",
        "benzoic acid": "benzohydrazide",
        "oxalic acid": "oxalohydrazide",
    }
    if name in retained:
        return retained[name]
    if name.endswith("benzoic acid"):
        return name.removesuffix("benzoic acid") + "benzohydrazide"
    for ending, suffix in (("sulfonic acid", "sulfonohydrazide"),
                           ("sulfinic acid", "sulfinohydrazide")):
        if name.endswith(ending):
            return name[:-len(ending)] + suffix
    if name.endswith("oic acid"):
        return name[:-len("oic acid")] + "ehydrazide"
    if name.endswith("carboxylic acid"):
        return name[:-len("carboxylic acid")] + "carbohydrazide"
    return ""


def _mono_hydrazide_proposals(mol, Chem, groups=None):
    """Yield mapped proposals from acid-capped parent constructions.

    Parent choice is made on a temporary acid with stereochemistry removed.
    The typed map is translated to source atom indices and decorated against
    the original graph, so original CIP priorities and isotope positions are
    retained. Principal-group comparison with other routes remains the
    caller's responsibility.
    """
    from .cross_route_selection import cross_route_selection

    sites = recognize_hydrazides(mol, annular_terminal=True)
    if len(sites) != 1:
        return
    if _potential_hydrazide_centers(mol) != frozenset({sites[0].center}):
        return  # N-acylated or otherwise nonordinary neighboring sites defer.
    site = sites[0]
    if groups is not None:
        selected_groups = getattr(groups, "principal", groups)
        if isinstance(selected_groups, HydrazideSite) or hasattr(selected_groups, "center"):
            selected_groups = (selected_groups,)
        if {getattr(g, "center", None) for g in selected_groups} != {site.center}:
            return
    from .extended_inventory import selection_inventory
    source_inventory = selection_inventory(mol)
    site_group = next((g for g in source_inventory.groups
                       if g.kind == "hydrazide" and g.center == site.center), None)
    if (not source_inventory.supported or site_group is None
            or any(g.kind in {"imide", "imine"} for g in source_inventory.groups)
            or any(g.kind != "hydrazide" and g.seniority < (12, 0)
                   for g in source_inventory.groups)):
        return
    if any(a.GetIsotope() and a.GetAtomicNum() in {7, 8} for a in mol.GetAtoms()):
        return  # N/O isotope role in the hydrazide characteristic unit is not yet proved.
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag())
           for i in (site.acyl_nitrogen, site.terminal_nitrogen)):
        return
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                    for a in mol.GetAtoms()):
        return
    # Remove both N atoms and their organyl branches from the temporary parent
    # graph. Those branches are named later from the original molecule.
    n_atoms = {site.acyl_nitrogen, site.terminal_nitrogen}
    substituent_atoms = set()
    for nitrogen, blocked in ((site.acyl_nitrogen, n_atoms | {site.center}),
                              (site.terminal_nitrogen, n_atoms)):
        roots = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() not in blocked]
        for root in roots:
            stack, branch = [root], set()
            while stack:
                idx = stack.pop()
                if idx in branch or idx in n_atoms:
                    continue
                branch.add(idx)
                stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                             if a.GetIdx() not in n_atoms and a.GetIdx() not in branch)
            attachments = {a.GetIdx() for idx in branch
                           for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if a.GetIdx() in n_atoms}
            if attachments != {nitrogen} or any(
                a.GetIdx() not in branch | n_atoms
                for idx in branch for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                if a.GetAtomicNum() > 1
            ):
                return
            substituent_atoms.update(branch)
    remove_atoms = n_atoms | substituent_atoms
    keep = {a.GetIdx() for a in mol.GetAtoms()} - remove_atoms
    order = sorted(keep)
    rw = Chem.RWMol(mol)
    for idx in sorted(remove_atoms, reverse=True):
        rw.RemoveAtom(idx)
    cap = Chem.Atom(8)
    cap_idx = rw.AddAtom(cap)
    rw.AddBond(order.index(site.center), cap_idx, Chem.BondType.SINGLE)
    acid = rw.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in acid.GetAtoms()):
        return
    parent_graph = Chem.Mol(acid)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    from .acid_derivative_methods import _capped_acid_selection
    selection = _capped_acid_selection(
        Chem, parent_graph.ToBinary(Chem.PropertyPickleOptions.AllProps),
        cross_route_selection)
    target_center = order.index(site.center)
    source_element = mol.GetAtomWithIdx(site.center).GetAtomicNum()
    acid_kind = ("acid" if source_element == 6 else
                 "sulfonic-acid" if len(site.atoms) == 5 else "sulfinic-acid")
    if selection is not None:
        selected = selection.selected
        if (selected.principal_kind != acid_kind
                or selected.principal_centers != frozenset({target_center})):
            return
    elif (mol.GetAtomWithIdx(site.terminal_nitrogen).IsInRing()
            or source_element == 6 and site_group.owner >= 0
            and mol.GetAtomWithIdx(site_group.owner).IsInRing()):
        from .mapped_ring_parents import records

        rows = records(parent_graph, Chem, general_branches=True)
        if len(rows) != 1 or rows[0].principal_centers != {target_center}:
            return
        selected = rows[0]
    else:
        return
    construction = selected.construction
    if construction is None or not construction.parent.endswith("acid"):
        return
    parent_name = _replace_acid_ending(construction.parent)
    if not parent_name:
        return
    locants = tuple((order[i], loc) for i, loc in construction.atom_locants if i < len(order))
    parent_atoms = frozenset(order[i] for i in construction.atoms if i < len(order))
    embedded = frozenset(order[i] for i in construction.embedded_atoms if i < len(order))
    if site.center in embedded and mol.GetAtomWithIdx(site.center).GetIsotope():
        return  # No locant has been proved for an external carbohydrazide carbonyl carbon.
    mapping = dict(locants)
    if set(parent_atoms) != set(mapping):
        return
    # P-66.3.3 gives N to the carbonyl-bound atom and N′ to the terminal atom.
    n_locants = ((site.acyl_nitrogen, "N"), (site.terminal_nitrogen, "N'"))
    n_prefixes, branch_atoms = [], set()
    from .component_mapping import MappedComponent, decorate_component, mapped_branch, mapped_general_branch
    for nitrogen, locant in n_locants:
        other_n = site.terminal_nitrogen if nitrogen == site.acyl_nitrogen else site.acyl_nitrogen
        if mol.GetAtomWithIdx(nitrogen).IsInRing():
            # A ring-closing terminal N shares both organyl bonds with a
            # single heterocycle. Cite the complete rooted N-yl as an amide
            # N-substituent (the terminal N belongs to that ring).
            from .component_mapping import mapped_general_branch

            text, owned = mapped_general_branch(mol, Chem, nitrogen, other_n)
            if not text or owned != substituent_atoms | {nitrogen}:
                return
            n_prefixes.append(('N', text))
            branch_atoms.update(owned)
            endings = {'carboxylic acid': 'carboxamide', 'oic acid': 'amide', 'ic acid': 'amide'}
            parent_name = next((construction.parent.removesuffix(ending) + amide
                                for ending, amide in endings.items()
                                if construction.parent.endswith(ending)), '')
            if not parent_name:
                return
            continue
        blocked = {site.center, other_n} if nitrogen == site.acyl_nitrogen else {other_n}
        roots = [a.GetIdx() for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                 if a.GetAtomicNum() > 1 and a.GetIdx() not in blocked]
        for root in roots:
            branch, owned = mapped_branch(mol, Chem, root, nitrogen)
            if (not branch or owned & (set(parent_atoms) | set(embedded) | branch_atoms
                                       | set(site.atoms))):
                return
            n_prefixes.append((locant, branch))
            branch_atoms.update(owned)
    # Rebuild acid-parent branches on the original graph too. The capped graph
    # can change CIP priorities at a stereocenter next to the derivative site.
    parent_prefixes, parent_branch_atoms = [], set()
    suffix_atoms = set(site.atoms)
    for parent_idx, locant in sorted(locants, key=lambda item: (item[1], item[0])):
        for atom in mol.GetAtomWithIdx(parent_idx).GetNeighbors():
            root = atom.GetIdx()
            if (atom.GetAtomicNum() <= 1 or root in parent_atoms or root in suffix_atoms
                    or root in branch_atoms or root in parent_branch_atoms):
                continue
            branch, owned = mapped_general_branch(mol, Chem, root, parent_idx)
            if (not branch or owned & (set(parent_atoms) | branch_atoms
                                       | parent_branch_atoms | suffix_atoms)):
                return
            parent_prefixes.append((locant, branch))
            parent_branch_atoms.update(owned)
    atom_locants = dict(locants)
    atom_locants.update((idx, loc) for idx, loc in n_locants if idx not in branch_atoms)
    component_atoms = frozenset(set(parent_atoms) | (n_atoms - branch_atoms))
    prefixes = tuple(parent_prefixes) + tuple(n_prefixes)
    # Every atom outside the mapped parent/suffix atoms is owned by one of
    # the rendered branches.  Keep that ownership on the component itself so
    # downstream composition can prove disjointness without reconstructing
    # branch traversal from the prefix text.
    # Carbonyl C/O atoms are part of the characteristic suffix even when a
    # ring parent construction embeds the external carboxyl unit. Give those
    # original atoms explicit ownership with an empty locant; the acid
    # construction's artificial cap is not an original atom.
    external_suffix_atoms = set()
    external_suffix_locants = {}
    if site.center not in parent_atoms:
        external_suffix_atoms.update(set(site.atoms) - n_atoms)
        external_suffix_locants.update({idx: "" for idx in external_suffix_atoms})
    embedded_atoms = frozenset((set(embedded) - external_suffix_atoms)
                               | branch_atoms | parent_branch_atoms)
    component_atoms = frozenset(set(component_atoms) | external_suffix_atoms)
    atom_locants.update(external_suffix_locants)
    if component_atoms & embedded_atoms:
        return
    component = MappedComponent(
        parent_name, tuple(atom_locants.items()), component_atoms, prefixes,
        embedded_atoms, construction.hydro_prefix, construction.systematic_parent,
        True, construction.isotope_unlocanted_elements, construction.prefix_mode,
        frozenset(order[i] for i in construction.local_isotope_atoms
                  if i < len(order) and order[i] in component_atoms),
        construction.omit_methyl_prefix_locant)
    name = decorate_component(mol, Chem, component)
    if not name:
        return
    required = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope()}
    covered = set(component_atoms) | set(embedded_atoms)
    # A characteristic carbonyl oxygen is included when the acid construction
    # assigns it a typed locant; opaque suffix-oxygen isotope roles defer.
    covered.update(external_suffix_atoms)
    if covered != required:
        return
    selected_path = tuple(order[i] for i in selected.parent_atoms if i < len(order))
    parent_mapping = dict(atom_locants)
    if not selected_path or any(i not in parent_mapping for i in selected_path):
        return
    # An external carbohydrazide carbonyl has no parent locant.  Its suffix
    # attaches at the selected path atom (commonly ring locant 1), rather than
    # at the artificial C1 used to construct the acid parent.
    suffix_atom = site.center if site.center in selected_path else site_group.owner
    if suffix_atom not in parent_mapping:
        return
    parent_kind = ("ring" if all(mol.GetAtomWithIdx(i).IsInRing() for i in selected_path)
                   else "chain")
    from .parent_selection import graph_parent_facts
    facts = graph_parent_facts(
        mol, selected_path, parent_mapping, principal_class=site_group.seniority,
        suffix_centers=(site.center,), suffix_locants=(parent_mapping[suffix_atom],),
        prefixes=prefixes, kind=parent_kind, name=name, allow_annotations=True)
    if facts is None:
        return
    yield HydrazideProposal(name, site, selected_path, tuple(atom_locants.items()),
                            n_locants, facts, prefixes, embedded_atoms, True, component)


def _poly_acid_ending(acid_parent, site_count, sites_on_path):
    """Translate only the proved di-/triacid endings required by P-66.3."""
    if acid_parent == "oxalic acid" and site_count == 2 and sites_on_path:
        return "oxalohydrazide"
    if sites_on_path == site_count:
        endings = {
            2: ("dioic acid", "dihydrazide"),
            3: ("trioic acid", "trihydrazide"),
        }
        ending = endings.get(site_count)
        if ending and acid_parent.endswith(ending[0]):
            return acid_parent[:-len(ending[0])] + ending[1]
    if sites_on_path == 0:
        endings = {
            2: ("dicarboxylic acid", "dicarbohydrazide"),
            3: ("tricarboxylic acid", "tricarbohydrazide"),
        }
        ending = endings.get(site_count)
        if ending and acid_parent.endswith(ending[0]):
            return acid_parent[:-len(ending[0])] + ending[1]
    return ""


def _symmetry_numbering_variants(mol, candidate, Chem, max_matches=512):
    """Map every complete-graph symmetry to an equivalent parent numbering.

    The shared acid selector can retain only one orientation when an
    aromatic/acyclic parent has a symmetry tie. Its selected construction is
    therefore expanded through actual automorphisms of the capped acid graph
    before hydrazide N/N′ locants are compared. The supplied graph has isotope
    labels cleared for this enumeration, in accord with P-82.5; mapped facts
    and final rendering still use the original isotopic molecule. This is a
    graph-derived set, not a site-order heuristic.
    """
    construction = candidate.construction
    if construction is None or not construction.atoms:
        return ()
    try:
        automorphisms = mol.GetSubstructMatches(
            mol, uniquify=False, useChirality=False, maxMatches=max_matches)
    except (RuntimeError, ValueError):
        return None  # The complete numbering frontier could not be proved.
    if len(automorphisms) >= max_matches:
        # RDKit may have truncated the match set at the requested bound. Do
        # not emit the selector's original orientation as if enumeration were
        # complete; an omitted orientation could win on N/N′ locants.
        return None
    variants = []
    for mapping in automorphisms:
        if all(index == mapped for index, mapped in enumerate(mapping)):
            continue
        if len(mapping) != mol.GetNumAtoms():
            continue
        new_locants = {mapping[index]: locant
                       for index, locant in construction.atom_locants}
        # Prefix roots have locants rather than atom maps in MappedComponent.
        # Recover the root from the old complete parent map and transfer the
        # prefix to that same source atom's new locant.
        old_locant_atoms = {}
        ambiguous = set()
        for atom_index, locant in construction.atom_locants:
            if locant in old_locant_atoms:
                ambiguous.add(locant)
            old_locant_atoms[locant] = atom_index
        prefixes = []
        for locant, text in construction.prefixes:
            if locant not in old_locant_atoms or locant in ambiguous:
                prefixes = None
                break
            root = old_locant_atoms[locant]
            prefixes.append((new_locants.get(root, locant), text))
        if prefixes is None:
            continue
        descriptor_parts = []
        for entry in construction.isotope_descriptor_parts:
            symbol, mass, locant, *tail = entry
            if locant not in old_locant_atoms or locant in ambiguous:
                descriptor_parts = None
                break
            root = old_locant_atoms[locant]
            descriptor_parts.append((symbol, mass, new_locants.get(root, locant), *tail))
        if descriptor_parts is None:
            continue
        try:
            mapped_construction = replace(
                construction,
                atom_locants=tuple(new_locants.items()),
                atoms=frozenset(mapping[index] for index in construction.atoms),
                embedded_atoms=frozenset(mapping[index]
                                          for index in construction.embedded_atoms),
                prefixes=tuple(prefixes),
                local_isotope_atoms=frozenset(
                    mapping[index] for index in construction.local_isotope_atoms),
                isotope_hydrogen_context_atoms=frozenset(
                    mapping[index] for index in construction.isotope_hydrogen_context_atoms),
                isotope_descriptor_parts=tuple(descriptor_parts),
            )
            variant = replace(
                candidate,
                principal_centers=frozenset(mapping[index]
                                             for index in candidate.principal_centers),
                parent_atoms=tuple(mapping[index] for index in candidate.parent_atoms),
                atom_locants=tuple((mapping[index], locant)
                                   for index, locant in candidate.atom_locants),
                construction=mapped_construction,
            )
        except (IndexError, TypeError, ValueError):
            continue
        variants.append(variant)
    return tuple(variants)


def _poly_hydrazide_proposals(mol, Chem, sites, groups=None):
    """Build one shared mapped acid parent for two or three hydrazide sites."""
    from .cross_route_selection import cross_route_selection
    from .extended_inventory import selection_inventory
    from .component_mapping import (MappedComponent, decorate_component,
                                    mapped_branch)

    if (len(sites) not in {2, 3}
            or any(mol.GetAtomWithIdx(site.center).GetAtomicNum() != 6 for site in sites)):
        return
    site_centers = frozenset(site.center for site in sites)
    if groups is not None:
        selected_groups = getattr(groups, "principal", groups)
        if hasattr(selected_groups, "center"):
            selected_groups = (selected_groups,)
        if frozenset(getattr(group, "center", None)
                     for group in selected_groups) != site_centers:
            return
    source_inventory = selection_inventory(mol)
    principal = tuple(source_inventory.principal)
    if (not source_inventory.supported or len(principal) != len(sites)
            or frozenset(group.center for group in principal) != site_centers
            or any(group.kind != "hydrazide" for group in principal)
            or _potential_hydrazide_centers(mol) != site_centers
            or any(group.kind in {"imide", "imine"}
                   for group in source_inventory.groups)
            or any(group.kind != "hydrazide" and group.seniority < (12, 0)
                   for group in source_inventory.groups)):
        return
    if (any(atom.GetIsotope() and atom.GetAtomicNum() in {7, 8}
            for atom in mol.GetAtoms())
            or any(int(mol.GetAtomWithIdx(index).GetChiralTag())
                   for site in sites
                   for index in (site.acyl_nitrogen, site.terminal_nitrogen))
            or mol.GetStereoGroups()
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                   for atom in mol.GetAtoms())):
        return

    # Remove all hydrazide nitrogens and their N-bound organyl components from
    # one graph. Branches that reconnect to the shared skeleton or another N
    # are not ordinary independently nameable substituents.
    n_atoms = {index for site in sites
               for index in (site.acyl_nitrogen, site.terminal_nitrogen)}
    suffix_atoms = {index for site in sites
                    for index in (site.center, site.carbonyl_oxygen,
                                  site.acyl_nitrogen, site.terminal_nitrogen)}
    substituent_atoms = set()
    for site in sites:
        for nitrogen, other, blocked in (
            (site.acyl_nitrogen, site.terminal_nitrogen,
             n_atoms | {site.center}),
            (site.terminal_nitrogen, site.acyl_nitrogen, n_atoms),
        ):
            roots = [atom.GetIdx() for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                     if atom.GetAtomicNum() > 1 and atom.GetIdx() not in blocked]
            for root in roots:
                stack, branch = [root], set()
                while stack:
                    idx = stack.pop()
                    if idx in branch or idx in n_atoms:
                        continue
                    branch.add(idx)
                    stack.extend(atom.GetIdx() for atom in mol.GetAtomWithIdx(idx).GetNeighbors()
                                 if atom.GetIdx() not in n_atoms and atom.GetIdx() not in branch)
                attachments = {atom.GetIdx() for idx in branch
                               for atom in mol.GetAtomWithIdx(idx).GetNeighbors()
                               if atom.GetIdx() in n_atoms}
                if (attachments != {nitrogen} or branch & substituent_atoms
                        or any(atom.GetIdx() not in branch | n_atoms
                               for idx in branch
                               for atom in mol.GetAtomWithIdx(idx).GetNeighbors()
                               if atom.GetAtomicNum() > 1)):
                    return
                substituent_atoms.update(branch)

    remove_atoms = n_atoms | substituent_atoms
    order = sorted(set(range(mol.GetNumAtoms())) - remove_atoms)
    rw = Chem.RWMol(mol)
    for idx in sorted(remove_atoms, reverse=True):
        rw.RemoveAtom(idx)
    target_centers = set()
    for site in sites:
        center_index = order.index(site.center)
        target_centers.add(center_index)
        cap = Chem.Atom(8)
        cap_index = rw.AddAtom(cap)
        rw.AddBond(center_index, cap_index, Chem.BondType.SINGLE)
    acid = rw.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return
    parent_graph = Chem.Mol(acid)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    numbering_graph = Chem.Mol(parent_graph)
    for atom in numbering_graph.GetAtoms():
        atom.SetIsotope(0)
    from .acid_derivative_methods import _capped_acid_selection
    selection = _capped_acid_selection(
        Chem, parent_graph.ToBinary(Chem.PropertyPickleOptions.AllProps),
        cross_route_selection)
    from types import SimpleNamespace
    if selection is None:
        from .cross_route_selection import ParentProposal
        from .mapped_ring_parents import parent_facts, records

        acid_groups = selection_inventory(parent_graph).principal
        candidates = []
        if acid_groups and all(group.kind == "acid" for group in acid_groups):
            for row in records(parent_graph, Chem, principal=acid_groups, general_branches=True):
                if (row.principal_centers != target_centers or row.construction is None
                        or not row.hydrogen_budget_proved or not row.annotation_coverage):
                    continue
                facts = parent_facts(parent_graph, row, acid_groups[0].seniority)
                if facts is not None:
                    candidates.append(ParentProposal(
                        row.name, "mapped-ring:" + row.method, "acid", row.principal_centers,
                        row.parent_atoms, row.atom_locants, facts, True, ring_state=row,
                        construction=row.construction))
        if len({row.name for row in candidates}) == 1:
            selection = SimpleNamespace(selected=candidates[0], candidates=candidates)
    if selection is None:
        return
    # Numbering alternatives are compared only when they describe this same
    # selected acid skeleton and systematic parent. Hydrazide N/N′ prefixes
    # are rebuilt for each mapped candidate before the shared comparator runs.
    selected_parent = selection.selected.construction
    selected_atoms = frozenset(selection.selected.parent_atoms)
    equivalents = [candidate for candidate in selection.candidates
                   if candidate.principal_kind == "acid"
                   and candidate.principal_centers == frozenset(target_centers)
                   and frozenset(candidate.parent_atoms) == selected_atoms
                   and candidate.construction is not None
                   and candidate.construction.parent == selected_parent.parent
                   and candidate.construction.systematic_parent == selected_parent.systematic_parent]
    if not equivalents:
        equivalents = [selection.selected]
    expanded = []
    seen_numberings = set()
    for candidate in equivalents:
        variants = _symmetry_numbering_variants(
            numbering_graph, candidate, Chem)
        if variants is None:
            return
        for variant in (candidate, *variants):
            construction_key = (
                variant.construction.parent,
                tuple(variant.parent_atoms),
                tuple(variant.construction.atom_locants),
            )
            if construction_key not in seen_numberings:
                seen_numberings.add(construction_key)
                expanded.append(variant)
    proposals = []
    for candidate in expanded:
        candidate_selection = SimpleNamespace(selected=candidate)
        proposal = next(_poly_candidate_proposals(
            mol, Chem, sites, source_inventory, candidate_selection,
            order, target_centers, n_atoms, suffix_atoms), None)
        if proposal is not None:
            proposals.append(proposal)
    if not proposals:
        return
    from .parent_selection import parent_frontier
    frontier = parent_frontier(proposals, lambda proposal: proposal.parent_selection)
    if len(frontier) == 1:
        yield frontier[0]
    elif frontier and len({proposal.name for proposal in frontier}) == 1:
        # Equivalent atom mappings can remain tied for a constitutionally
        # symmetric parent. Since they compose to the same PIN, emit one of
        # the proved tied proposals rather than treating graph traversal order
        # as a nomenclatural preference.
        yield frontier[0]


def _poly_candidate_proposals(mol, Chem, sites, source_inventory, selection,
                              order, target_centers, n_atoms, suffix_atoms):
    from .component_mapping import (MappedComponent, decorate_component,
                                    mapped_branch)
    principal = tuple(source_inventory.principal)
    if (selection.selected.principal_kind != "acid"
            or selection.selected.principal_centers != frozenset(target_centers)):
        return
    construction = selection.selected.construction
    if construction is None or not construction.parent.endswith("acid"):
        return
    selected_path = tuple(order[index] for index in selection.selected.parent_atoms
                          if index < len(order))
    if not selected_path:
        return
    selected_path_set = set(selected_path)
    sites_on_path = sum(site.center in selected_path_set for site in sites)
    parent_name = _poly_acid_ending(construction.parent, len(sites), sites_on_path)
    if not parent_name:
        return

    # Translate all shared acid-parent atoms and locants to original IDs.
    locants = tuple((order[index], locant)
                    for index, locant in construction.atom_locants
                    if index < len(order))
    parent_atoms = frozenset(order[index] for index in construction.atoms
                             if index < len(order))
    embedded = frozenset(order[index] for index in construction.embedded_atoms
                         if index < len(order))
    parent_mapping = dict(locants)
    if set(parent_atoms) != set(parent_mapping) or parent_atoms & embedded:
        return
    if any(site.center in embedded and mol.GetAtomWithIdx(site.center).GetIsotope()
           for site in sites):
        return  # External carbonyl-carbon isotope locants are not yet proved.

    group_by_center = {group.center: group for group in principal}
    site_locants = []
    n_locants = []
    for site in sites:
        attachment_atom = (site.center if site.center in selected_path_set
                           else group_by_center[site.center].owner)
        if attachment_atom not in parent_mapping:
            return
        locant = parent_mapping[attachment_atom]
        if not locant.isdigit():
            return  # Superscript parent-site locants require a numeric locant.
        site_locants.append((site.center, locant))
        n_locants.extend(((site.acyl_nitrogen, f"N{locant}"),
                          (site.terminal_nitrogen, f"N'{locant}")))

    n_prefixes = []
    n_branch_atoms = set()
    for site, (_, site_locant) in zip(sites, site_locants, strict=True):
        for nitrogen, other, nitrogen_locant in (
            (site.acyl_nitrogen, site.terminal_nitrogen, f"N{site_locant}"),
            (site.terminal_nitrogen, site.acyl_nitrogen, f"N'{site_locant}"),
        ):
            blocked = (n_atoms | {site.center} if nitrogen == site.acyl_nitrogen
                       else n_atoms)
            roots = [atom.GetIdx() for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                     if atom.GetAtomicNum() > 1 and atom.GetIdx() not in blocked]
            for root in roots:
                branch, owned = mapped_branch(mol, Chem, root, nitrogen)
                if (not branch or owned & (set(parent_atoms) | set(embedded)
                                           | n_branch_atoms | suffix_atoms)):
                    return
                n_prefixes.append((nitrogen_locant, branch))
                n_branch_atoms.update(owned)

    parent_prefixes = []
    parent_branch_atoms = set()
    for parent_idx, locant in sorted(locants, key=lambda item: (item[1], item[0])):
        for atom in mol.GetAtomWithIdx(parent_idx).GetNeighbors():
            root = atom.GetIdx()
            if (atom.GetAtomicNum() <= 1 or root in parent_atoms
                    or root in suffix_atoms or root in n_branch_atoms
                    or root in parent_branch_atoms):
                continue
            branch, owned = mapped_branch(mol, Chem, root, parent_idx)
            if (not branch or owned & (set(parent_atoms) | n_branch_atoms
                                       | parent_branch_atoms | suffix_atoms)):
                return
            parent_prefixes.append((locant, branch))
            parent_branch_atoms.update(owned)

    all_locants = dict(locants)
    all_locants.update(n_locants)
    external_suffix_atoms = {
        atom_index
        for site in sites if site.center not in parent_atoms
        for atom_index in (site.center, site.carbonyl_oxygen)
    }
    all_locants.update({atom_index: "" for atom_index in external_suffix_atoms})
    component_atoms = frozenset(set(parent_atoms) | n_atoms | external_suffix_atoms)
    prefixes = tuple(parent_prefixes) + tuple(n_prefixes)
    embedded_atoms = frozenset((set(embedded) - external_suffix_atoms)
                               | n_branch_atoms | parent_branch_atoms)
    if component_atoms & embedded_atoms:
        return
    component = MappedComponent(
        parent_name, tuple(all_locants.items()), component_atoms, prefixes,
        embedded_atoms, construction.hydro_prefix, construction.systematic_parent,
        True, construction.isotope_unlocanted_elements, construction.prefix_mode,
        frozenset(order[index] for index in construction.local_isotope_atoms
                  if index < len(order) and order[index] in component_atoms),
        construction.omit_methyl_prefix_locant)
    # Multiple hydrazide units need superscripted N/N′ site locants in both
    # nitrogen substitutions and atom-bound isotope/stereo annotations.
    component = MappedComponent(
        component.parent, component.atom_locants, component.atoms, component.prefixes,
        component.embedded_atoms, component.hydro_prefix, component.systematic_parent,
        True, component.isotope_unlocanted_elements, component.prefix_mode,
        component.local_isotope_atoms, component.omit_methyl_prefix_locant,
        component.isotope_descriptor_parts)
    name = decorate_component(mol, Chem, component)
    if not name:
        return
    required = {atom.GetIdx() for atom in mol.GetAtoms()
                if atom.GetAtomicNum() > 1 or atom.GetIsotope()}
    if set(component_atoms) | set(embedded_atoms) != required:
        return
    if component_atoms & embedded_atoms:
        return
    from .parent_selection import graph_parent_facts
    suffix_locants = tuple(parent_mapping[site_locant_owner]
                           for site_locant_owner in
                           (site.center if site.center in selected_path_set
                            else group_by_center[site.center].owner for site in sites))
    cyclic = all(mol.GetAtomWithIdx(index).IsInRing() for index in selected_path)
    facts = graph_parent_facts(
        mol, selected_path, all_locants,
        principal_class=principal[0].seniority,
        suffix_centers=tuple(site.center for site in sites),
        suffix_locants=suffix_locants, prefixes=prefixes,
        kind="ring" if cyclic else "chain", name=name, allow_annotations=True)
    if facts is None:
        return
    yield PolyHydrazideProposal(
        name, tuple(sites), tuple(site_locants), selected_path,
        tuple(all_locants.items()), tuple(n_locants), facts, prefixes,
        embedded_atoms, True, component)


def hydrazide_proposals(mol, Chem, groups=None):
    """Yield mono-site or shared multi-site acid-derived proposals.

    Multiple sites are selected together on one acid-capped graph, then all
    original atom maps are reconstructed before the parent proposal adapter
    is used by cross-route selection.
    """
    sites = recognize_hydrazides(mol, annular_terminal=True)
    if len(sites) == 1:
        yield from _mono_hydrazide_proposals(mol, Chem, groups)
    elif len(sites) > 1 and not any(mol.GetAtomWithIdx(site.terminal_nitrogen).IsInRing()
                                    for site in sites):
        yield from _poly_hydrazide_proposals(mol, Chem, sites, groups)
