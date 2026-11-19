"""Mapped, bounded P-62.5 functional-class names for single imine oxides.

The source graph is the authority for annotations. A private neutral imine copy
is used only to ask the established P-62.3 chain renderer for candidate parent
constructions; those constructions are remapped to the original atom indices
and decorated against the original C=N–O graph.
"""

from dataclasses import dataclass

RULE = "P-41/P-44/P-62.3/P-62.5(1)/P-74.2.2.1.9(3)/P-82/P-91.2.1.2.1"


@dataclass(frozen=True, slots=True)
class ImineOxideSite:
    carbon: int
    nitrogen: int
    oxygen: int
    n_substituent: int

    @property
    def atoms(self):
        return frozenset((self.carbon, self.nitrogen, self.oxygen))


def recognize_sites(mol):
    """Recognize localized C=N+(O−) imine oxides with one N-organyl arm.

    This intentionally excludes N–H compounds (P-74.2.1.2.9 says those are
    outside the nitrone class), amine oxides, nitro compounds, oximes, and
    multiply oxidized or multiply imine structures.
    """
    found = []
    for n in mol.GetAtoms():
        if (n.GetAtomicNum() != 7 or n.GetFormalCharge() != 1
                or n.GetNumRadicalElectrons() or n.IsInRing()):
            continue
        doubles = [a for a in n.GetNeighbors()
                   if a.GetAtomicNum() == 6
                   and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        oxides = [a for a in n.GetNeighbors()
                  if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1
                  and mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(doubles) != 1 or len(oxides) != 1:
            continue
        c, o = doubles[0], oxides[0]
        other = [a for a in n.GetNeighbors() if a.GetIdx() not in {c.GetIdx(), o.GetIdx()}
                 and a.GetAtomicNum() > 1]
    # P-74.2.2.1.9 excludes R2C=N+(O−)H from the nitrone class. We bound
        # this producer to carbon substitution on N; other heteroatom
        # substituents have their own class competition.
        if (len(other) != 1 or other[0].GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(n.GetIdx(), other[0].GetIdx()).GetBondTypeAsDouble() != 1
                or o.GetDegree() != 1 or o.GetTotalNumHs(includeNeighbors=True)
                or n.GetTotalValence() != 4 or c.GetFormalCharge()
                or c.GetNumRadicalElectrons()):
            continue
        found.append(ImineOxideSite(c.GetIdx(), n.GetIdx(), o.GetIdx(), other[0].GetIdx()))
    return tuple(found)


def _eligible_original(mol, site):
    """Bound source graph to one carbon-only imine parent and N-alkyl arm."""
    if mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups():
        return False
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if any(a.GetAtomicNum() not in {1, 6, 7, 8} or a.GetNumRadicalElectrons()
           for a in mol.GetAtoms()):
        return False
    if any(a.GetAtomicNum() == 1 and (a.GetDegree() != 1
           or a.GetNeighbors()[0].GetAtomicNum() == 1) for a in mol.GetAtoms()):
        return False
    if {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7} != {site.nitrogen}:
        return False
    if {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8} != {site.oxygen}:
        return False
    # Core N/O isotope and isotope-H writing has not been established for the
    # functional-class N-oxide grammar. Carbon isotopes remain mapped below.
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in (site.nitrogen, site.oxygen)):
        return False
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return False
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in (site.nitrogen, site.oxygen)):
        return False
    # Remote atom stereo can change CIP ordering when N–O is removed from the
    # private parent-search copy. Until cross-copy stereo numbering is proved,
    # keep this path to C=N bond geometry only.
    if any(int(a.GetChiralTag()) for a in mol.GetAtoms()):
        return False
    if any(int(b.GetStereo()) and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()}
           != {site.carbon, site.nitrogen} for b in mol.GetBonds()):
        return False
    # A single connected component and exact C=N, N–O, and N–C core are
    # required. The remaining graph must consist only of carbon/hydrogen.
    if len(ChemComponents(mol)) != 1:
        return False
    if heavy != ({a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
                 | {site.nitrogen, site.oxygen}):
        return False
    return True


def ChemComponents(mol):
    """Tiny component finder independent of RDKit's optional helper APIs."""
    remaining = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, component = [seed], {seed}
        while stack:
            i = stack.pop()
            for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                j = atom.GetIdx()
                if j in remaining:
                    remaining.remove(j)
                    component.add(j)
                    stack.append(j)
        components.append(frozenset(component))
    return tuple(components)


def _neutral_imine_copy(mol, site, Chem):
    """Delete only N–O, neutralize N, and tag all surviving original IDs."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetProp("_nitrone_origin_idx", str(atom.GetIdx()))
    rw.RemoveBond(site.nitrogen, site.oxygen)
    rw.RemoveAtom(site.oxygen)
    # Removing a terminal N–O bond shifts later atom indices, but the private
    # property provides an exact bijection back to the caller's atom IDs.
    n_new = next(a.GetIdx() for a in rw.GetAtoms()
                 if int(a.GetProp("_nitrone_origin_idx")) == site.nitrogen)
    n = rw.GetAtomWithIdx(n_new)
    n.SetFormalCharge(0)
    # The capped C=N endpoint no longer has the original nitrone ligands.
    # Never carry its original E/Z tag or directional bond marks into the
    # neutral imine parent search; the final component is decorated on `mol`.
    c_new = next(a.GetIdx() for a in rw.GetAtoms()
                 if int(a.GetProp("_nitrone_origin_idx")) == site.carbon)
    double = rw.GetBondBetweenAtoms(c_new, n_new)
    double.SetStereo(Chem.BondStereo.STEREONONE)
    double.SetBondDir(Chem.BondDir.NONE)
    for atom_idx in (c_new, n_new):
        for bond in rw.GetAtomWithIdx(atom_idx).GetBonds():
            if bond.GetIdx() != double.GetIdx():
                bond.SetBondDir(Chem.BondDir.NONE)
    # The supported nitrone has an N-carbon substituent, so neutralization
    # gives the ordinary tertiary-valence C=N–C imine without adding H.
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _original_component(mol, Chem, construction, copy):
    from .component_mapping import MappedComponent, decorate_component

    origin = {a.GetIdx(): int(a.GetProp("_nitrone_origin_idx")) for a in copy.GetAtoms()}
    try:
        mapped = MappedComponent(
            construction.parent + " N-oxide",
            tuple((origin[i], loc) for i, loc in construction.atom_locants)
            + ((next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8), "N"),),
            frozenset(origin[i] for i in construction.atoms)
            | {next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)},
            tuple((loc, text) for loc, text in construction.prefixes),
            frozenset(origin[i] for i in construction.embedded_atoms),
            construction.hydro_prefix,
            construction.systematic_parent,
            construction.nitrogen_site_locants,
            construction.isotope_unlocanted_elements,
            construction.prefix_mode,
            frozenset(origin[i] for i in construction.local_isotope_atoms),
            construction.omit_methyl_prefix_locant,
            construction.isotope_descriptor_parts,
        )
    except (KeyError, StopIteration):
        return "", None
    name = decorate_component(mol, Chem, mapped)
    return (name, mapped) if name else ("", None)


def _construction_for_proposal(copy, Chem, inventory, proposal):
    """Recover the typed mapped component from the imine chain proposal.

    The existing imine producer predates the shared ``construction`` field.
    This recovery repeats its component assembly from the proposal's own
    mapped parent and locants, never from its rendered name.
    """
    from .component_mapping import MappedComponent, hydrogen_bindings, mapped_branch
    from .cross_route_selection import _annotation_ownership
    from .branch_rules import parent_chain_name
    from .formatting import complete_parent_name
    from .imine_components import imine_ending

    seq = tuple(proposal.parent_atoms)
    carbon_map = dict(proposal.atom_locants)
    if not seq or any(i not in carbon_map for i in seq):
        return None
    mapping = dict(carbon_map)
    prefixes, embedded, failed = [], set(), False
    for idx, locant in tuple(mapping.items()):
        atom = copy.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            root = neighbor.GetIdx()
            if (root in mapping or neighbor.GetAtomicNum() <= 1
                    or copy.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1):
                continue
            text, owned = mapped_branch(copy, Chem, root, idx)
            if not text or owned & (mapping.keys() | embedded) or any(c.isspace() for c in text):
                failed = True
                break
            prefixes.append((locant, text))
            embedded.update(owned)
        if failed:
            break
    mapping = hydrogen_bindings(copy, mapping)
    needed = {a.GetIdx() for a in copy.GetAtoms()
              if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())}
    if failed or set(mapping) | embedded != needed or not _annotation_ownership(copy, mapping):
        return None
    parent = complete_parent_name(parent_chain_name(copy, seq))
    groups = inventory.principal
    if not parent or not groups or any(g.center not in seq for g in groups):
        return None
    locants = tuple(sorted((str(carbon_map[g.center]) for g in groups), key=int))
    # P-82.6.1.1 restores every otherwise omitted locant when an isotope
    # position needs to be specified. In particular, a labelled C in a
    # two-carbon imine parent requires its suffix locant as well.
    parent_carbons = [i for i in seq if copy.GetAtomWithIdx(i).GetAtomicNum() == 6]
    isotope_position_required = any(copy.GetAtomWithIdx(i).GetIsotope() for i in parent_carbons) \
        and not (parent_carbons and all(copy.GetAtomWithIdx(i).GetIsotope()
                                         == copy.GetAtomWithIdx(parent_carbons[0]).GetIsotope()
                                         for i in parent_carbons))
    suffix_locants = (() if len(groups) == 1 and len(seq) <= 2
                      and not isotope_position_required else locants)
    ending, _ = imine_ending(copy, Chem, parent, groups, suffix_locants)
    if not ending:
        return None
    from .imine_components import unlocanted_imine_nitrogen
    return MappedComponent(
        ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
        frozenset(embedded), systematic_parent=True,
        nitrogen_site_locants=True,
        isotope_unlocanted_elements=unlocanted_imine_nitrogen(copy),
    )


def nitrone_decision(mol, Chem):
    """Return a complete method-1 decision for the bounded single-nitrone graph."""
    sites = recognize_sites(mol)
    if len(sites) != 1 or not _eligible_original(mol, sites[0]):
        return None
    site = sites[0]
    copy = _neutral_imine_copy(mol, site, Chem)
    from .component_mapping import chemical_inventory
    inventory = chemical_inventory(copy)
    if (not inventory.supported or len(inventory.groups) != 1
            or not inventory.principal or inventory.principal[0].kind != "imine"
            or inventory.principal[0].center != next(
                a.GetIdx() for a in copy.GetAtoms()
                if int(a.GetProp("_nitrone_origin_idx")) == site.carbon)):
        return None
    from .imine_components import chain_proposals
    proposals = list(chain_proposals(copy, Chem, inventory))
    if not proposals:
        return None
    from .parent_selection import parent_frontier
    frontier = parent_frontier(proposals, lambda p: p.parent_selection)
    decorated = []
    for proposal in frontier:
        construction = proposal.construction or _construction_for_proposal(
            copy, Chem, inventory, proposal)
        if construction is None:
            return None
        decorated.append(_original_component(mol, Chem, construction, copy))
    if not decorated or any(not name for name, _ in decorated):
        return None
    # Symmetry-equivalent maps may tie with the same PIN. Distinct spellings
    # from an unresolved parent frontier are preserved as a decline.
    names = {name for name, _ in decorated}
    if len(names) != 1:
        return None
    name = next(iter(names))
    mapped = decorated[0][1]
    parent_role = frozenset((set(mapped.atoms) | set(mapped.embedded_atoms)) - {site.oxygen})
    oxide_role = frozenset({site.oxygen})
    if parent_role & oxide_role or parent_role | oxide_role != frozenset(
            a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1):
        return None
    original_locants = tuple((i, loc) for i, loc in mapped.atom_locants if i in parent_role)
    from .naming_methods import MethodDecision
    return MethodDecision(
        name, "functional-class", frozenset({site.carbon}),
        frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
        RULE, mapped.atom_locants,
        component_roles=(("imine-parent", parent_role, original_locants),
                         ("N-oxide-oxygen", oxide_role, ())),
    )


def imine_oxide_decision(mol, Chem):
    """Alias following the naming-method producer convention."""
    return nitrone_decision(mol, Chem)
