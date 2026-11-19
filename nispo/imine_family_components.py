"""Unified mapped acyclic chain proposals for the class-20 imine family.

The site recognizers and local N-substituent operations remain in
``imine_components`` and ``oxime_components``. This module combines their
neutral, explicitly supported C=N sites under the shared imine suffix while
preserving original atom and annotation ownership.
"""

RULES = (
    "P-16.9.3",
    "P-41/P-43",
    "P-44.1.1/P-44.3/P-44.4/P-45.2",
    "P-62.3.1.1/P-62.4",
    "P-68.3.1.1.2",
    "P-82.2",
    "P-93.4.2.1.3",
)

_OXIME_KINDS = {"oxime", "oxime-ether"}


def _site(mol, group):
    """Return typed local site data, rejecting ambiguous N-ligand chemistry."""
    from .imine_components import ordinary_imine, imine_nitrogen
    from .oxime_components import oxime_site

    kind = getattr(group, "kind", None)
    carbon = getattr(group, "center", -1)
    if not 0 <= carbon < mol.GetNumAtoms():
        return None
    atom = mol.GetAtomWithIdx(carbon)
    if atom.IsInRing():
        return None
    double_bonds = [bond for bond in atom.GetBonds()
                    if bond.GetBondTypeAsDouble() == 2]
    if (len(double_bonds) != 1
            or mol.GetAtomWithIdx(double_bonds[0].GetOtherAtomIdx(carbon)).GetAtomicNum() != 7
            or any(bond.GetBondTypeAsDouble() > 1 for bond in atom.GetBonds()
                   if bond.GetBondTypeAsDouble() != 2)):
        # A cumulated C=C=N unit is a ketenimine class, not an ordinary
        # C=N imine under this bounded family route.
        return None
    if kind == "imine":
        if not ordinary_imine(mol, carbon):
            return None
        nitrogen = imine_nitrogen(mol, carbon)
        return {"kind": kind, "carbon": carbon, "nitrogen": nitrogen,
                "oxygen": -1, "atoms": (carbon, nitrogen)}
    if kind in _OXIME_KINDS:
        site = oxime_site(mol, carbon)
        if site is None or site.kind != kind:
            return None
        return {"kind": kind, "carbon": carbon, "nitrogen": site.nitrogen,
                "oxygen": site.oxygen, "atoms": site.atoms}
    return None


def principal_scope(mol, groups):
    """Accept a complete set of supported neutral imine-family sites."""
    groups = tuple(groups)
    if not groups or any(_site(mol, group) is None for group in groups):
        return False
    centers = [group.center for group in groups]
    return len(centers) == len(set(centers))


def _stereo_descriptor(mol, Chem, site):
    from .stereochemistry import _indexed_bond_stereochemistry

    pair = {site["carbon"], site["nitrogen"]}
    return next((descriptor for a, b, descriptor in
                 _indexed_bond_stereochemistry(mol, Chem) if {a, b} == pair), "")


def _site_substituent(mol, Chem, site, root, blocked):
    """Return a mapped substituent prefix and the original atoms it owns."""
    if site["oxygen"] == root and site["nitrogen"] == blocked:
        from .oxime_components import _n_oxygen_substituent
        return _n_oxygen_substituent(mol, Chem, root, blocked)
    from .component_mapping import mapped_branch
    return mapped_branch(mol, Chem, root, blocked)


def chain_proposals(mol, Chem, inventory, *, principal=None, scope=None):
    """Yield complete atom-mapped chain candidates for mixed imine-family sites.

    The caller owns P-41/P-43 selection. ``principal`` is therefore the
    exact selected family tuple when supplied; otherwise the inventory's
    principal tuple is used. Ring parents and N-derivatives outside ordinary
    imines and simple oximes/oxime ethers are declined.
    """
    from .multifunctional_parent import _paths
    from .component_mapping import (MappedComponent, decorate_component,
                                    hydrogen_bindings, mapped_branch)
    from .formatting import attach_suffix, complete_parent_name, multiplied_suffix
    from .branch_rules import parent_chain_name
    from .parent_selection import graph_parent_facts
    from .cross_route_selection import ParentProposal, _annotation_ownership
    from .stereochemistry import assigned_atom_stereochemistry

    groups = tuple(inventory.principal if principal is None else principal)
    if not principal_scope(mol, groups):
        return
    sites = {group.center: _site(mol, group) for group in groups}
    if any(value is None for value in sites.values()):
        return
    carbon = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and not atom.IsInRing()
        and (scope is None or atom.GetIdx() in scope)
    }
    for path in _paths(mol, carbon):
        for seq in (path, tuple(reversed(path))):
            if not set(sites) <= set(seq):
                continue
            carbon_mapping = {idx: str(pos) for pos, idx in enumerate(seq, 1)}
            mapping = dict(carbon_mapping)
            for center, site in sites.items():
                mapping[site["nitrogen"]] = (
                    "N" + carbon_mapping[center] if len(groups) > 1 else "N"
                )

            prefixes, embedded, failed = [], set(), False
            # Site atoms are traversed from the mapped C/N atoms. The special
            # N-O operation owns O and its complete simple alkyl branch.
            for idx, locant in tuple(mapping.items()):
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root = neighbor.GetIdx()
                    if (root in mapping or neighbor.GetAtomicNum() <= 1
                            or (scope is not None and root not in scope)):
                        continue
                    bond = mol.GetBondBetweenAtoms(idx, root)
                    if bond is None or bond.GetBondTypeAsDouble() != 1:
                        failed = True
                        break
                    site = next((item for item in sites.values()
                                 if item["nitrogen"] == idx
                                 and item["oxygen"] == root), None)
                    text, owned = (_site_substituent(mol, Chem, site, root, idx)
                                   if site else mapped_branch(mol, Chem, root, idx))
                    if (not text or owned & (mapping.keys() | embedded)
                            or any(char.isspace() for char in text)):
                        failed = True
                        break
                    prefixes.append((locant, text))
                    embedded.update(owned)
                if failed:
                    break
            mapping = hydrogen_bindings(mol, mapping)
            needed = {atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1 or atom.GetIsotope()
                      or int(atom.GetChiralTag())}
            target = needed if scope is None else needed & set(scope)
            if (failed or set(mapping) | embedded != target
                    or not _annotation_ownership(mol, mapping)):
                continue

            parent = complete_parent_name(parent_chain_name(mol, seq))
            if not parent:
                continue
            locants = tuple(sorted((int(carbon_mapping[center]) for center in sites)))
            locants_text = tuple(str(loc) for loc in locants)
            suffix = multiplied_suffix(len(groups), "imine")
            if not suffix:
                continue
            suffix_locants = (() if len(groups) == 1 and len(seq) <= 2
                              else locants_text)
            ending = attach_suffix(parent, suffix, suffix_locants)
            if not ending:
                continue

            # Retain source geometry from the unmodified graph, then suppress
            # only C=N geometry on the private rendering copy so generic
            # alkene decoration cannot add a second descriptor.
            decoration_mol = Chem.Mol(mol)
            geometry = []
            for center, site in sites.items():
                descriptor = _stereo_descriptor(mol, Chem, site)
                if descriptor:
                    locant = carbon_mapping[center] if suffix_locants else ""
                    geometry.append((int(locant) if locant else 0,
                                     f"{locant}{descriptor}"))
                bond = decoration_mol.GetBondBetweenAtoms(center, site["nitrogen"])
                if bond is not None:
                    bond.SetStereo(Chem.BondStereo.STEREONONE)
            original_stereo = dict(assigned_atom_stereochemistry(
                mol, Chem, atom_order="idx", include_pseudo=True,
            ))
            copied_stereo = dict(assigned_atom_stereochemistry(
                decoration_mol, Chem, atom_order="idx", include_pseudo=True,
            ))
            if any(original_stereo.get(idx, "") != copied_stereo.get(idx, "")
                   for idx in original_stereo.keys() | copied_stereo.keys()):
                continue

            component = MappedComponent(
                ending, tuple(mapping.items()), frozenset(mapping),
                tuple(prefixes), frozenset(embedded), systematic_parent=True,
                nitrogen_site_locants=True,
            )
            name = decorate_component(decoration_mol, Chem, component)
            if not name or len(name) > 260:
                continue
            if geometry:
                geometry.sort()
                name = "(" + ",".join(text for _, text in geometry) + ")-" + name
            facts = graph_parent_facts(
                mol, seq, mapping, principal_class=groups[0].seniority,
                suffix_centers=tuple(sites), suffix_locants=locants,
                prefixes=prefixes, kind="chain", name=name,
                allow_annotations=True,
            )
            if facts is None:
                continue
            yield ParentProposal(
                name, "mapped-chain-imine-family", "imine",
                frozenset(sites), tuple(seq), tuple(mapping.items()), facts, True,
                construction=component,
            )
