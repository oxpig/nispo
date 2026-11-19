"""Mapped neutral carbon peroxols under Blue Book P-63.4.

Recognition records neutral C–X–Y–H operations with original atom indices,
where X and Y are the Table 6.1 O/S/Se/Te pairs. Peroxy acids and isotope
labels on the chalcogen bridge or terminal hydrogen remain outside this rule.
"""

from .molecule_iteration import iter_atoms, iter_bonds
from .functional_inventory import CharacteristicGroup

RULES = (
    "P-41-class-18",
    "P-44.1.1/P-45.2",
    "P-63.4.1", "P-63.4.2.1/P-63.4.2.2", "P-63.7",
    "P-16.7/P-16.9",
    "P-82.2/P-82.6.1.1", "P-14.3.4.4",
)

_SITE_TYPE_BY_PAIR = {
    (8, 8): "OOH",
    (16, 8): "SO", (34, 8): "SeO", (52, 8): "TeO",
    (8, 16): "OS", (8, 34): "OSe", (8, 52): "OTe",
    (16, 16): "SS", (34, 16): "SeS", (52, 16): "TeS",
    (16, 34): "SSe", (16, 52): "STe", (34, 34): "SeSe",
    (52, 34): "TeSe", (34, 52): "SeTe", (52, 52): "TeTe",
}
# Table 6.1 is ordered by reading down the left column, then down the right
# column. OOH retains the preceding hydroperoxide seniority.
_SITE_RANK_BY_TYPE = {
    "OOH": 0,
    "SO": 1, "SeO": 2, "TeO": 3,
    "OS": 4, "OSe": 5, "OTe": 6,
    "SS": 7,
    "SeS": 8, "TeS": 9, "SSe": 10, "STe": 11, "SeSe": 12,
    "TeSe": 13, "SeTe": 14, "TeTe": 15,
}


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def _ordinary_oxygen(atom):
    return (atom.GetAtomicNum() == 8 and not atom.GetFormalCharge()
            and not atom.GetNumRadicalElectrons() and not atom.IsInRing())


def _ordinary_chalcogen(atom, atomic_number):
    return (atom.GetAtomicNum() == atomic_number and not atom.GetFormalCharge()
            and not atom.GetNumRadicalElectrons() and not atom.IsInRing())


def recognize_peroxols(mol):
    """Return neutral C–X–Y–H peroxols (X,Y in O/S/Se/Te) with original IDs.

    `center` is the proximal oxygen as a suffix operation center, `owner` is
    the skeletal carbon that receives the suffix locant, and `ligand` is the
    terminal oxygen. This mirrors the group/owner split used by functional
    groups with a suffix atom outside their parent skeleton.
    """
    found = []
    allowed_pairs = {
        (8, 8),
        (16, 8), (34, 8), (52, 8),
        (8, 16), (8, 34), (8, 52),
        (16, 16), (34, 16), (52, 16),
        (16, 34), (16, 52), (34, 34), (52, 34), (34, 52), (52, 52),
    }
    for proximal in iter_atoms(mol):
        p = proximal.GetIdx()
        x = proximal.GetAtomicNum()
        if x not in {8, 16, 34, 52} or proximal.GetDegree() != 2:
            continue
        hosts = [a for a in proximal.GetNeighbors() if a.GetAtomicNum() == 6]
        terminals = [a for a in proximal.GetNeighbors()
                     if a.GetAtomicNum() in {8, 16, 34, 52}]
        if len(hosts) != 1 or len(terminals) != 1:
            continue
        host, terminal = hosts[0], terminals[0]
        h, t = host.GetIdx(), terminal.GetIdx()
        y = terminal.GetAtomicNum()
        if ((x, y) not in allowed_pairs
                or not _ordinary_chalcogen(proximal, x)
                or not _ordinary_chalcogen(terminal, y)
                or terminal.GetDegree() != 1
                or not _single(mol, h, p) or not _single(mol, p, t)
                or terminal.GetTotalNumHs(includeNeighbors=True) != 1
                or host.GetFormalCharge() or host.GetNumRadicalElectrons()):
            continue
        # The acid/peroxy-acid C(=O)OOH motif belongs to acid replacement
        # nomenclature, not to the carbon-bound peroxol suffix.
        if x == 8 and y == 8 and any(n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(h, n.GetIdx()).GetBondTypeAsDouble() == 2
               for n in host.GetNeighbors()):
            continue
        found.append(CharacteristicGroup(
            "peroxol", p, (p, t), h, t,
            _SITE_RANK_BY_TYPE[_SITE_TYPE_BY_PAIR[(x, y)]],
        ))
    return tuple(found)


def additional_sites(mol):
    """Extended-inventory adapter: groups, covered atoms, used N, supported."""
    groups = recognize_peroxols(mol)
    return groups, {i for group in groups for i in group.atoms}, set(), True


def principal_scope(mol, groups):
    """Admit only fully recognized neutral peroxol groups in the scope."""
    detected = {(g.center, g.owner, g.ligand) for g in recognize_peroxols(mol)}
    return bool(groups) and all(
        g.kind == "peroxol" and (g.center, g.owner, g.ligand) in detected
        for g in groups
    )


def suffix_roles(mol, group):
    """Return the mapped P-63.4 suffix roles, or an empty map if unsupported."""
    if not principal_scope(mol, (group,)):
        return {}
    elements = (mol.GetAtomWithIdx(group.center).GetAtomicNum(),
                mol.GetAtomWithIdx(group.ligand).GetAtomicNum())
    if elements == (8, 8):
        return {"host": group.owner, "proximal_oxygen": group.center,
                "terminal_oxygen": group.ligand}
    return {"host": group.owner, "proximal": group.center,
            "terminal": group.ligand,
            "proximal_chalcogen": group.center,
            "terminal_chalcogen": group.ligand}


def peroxol_site_type(mol, group):
    """Return the exact Table 6.1 atom-order key for a recognized group."""
    roles = suffix_roles(mol, group)
    if not roles:
        return None
    proximal_id = roles.get("proximal", roles.get("proximal_oxygen"))
    terminal_id = roles.get("terminal", roles.get("terminal_oxygen"))
    proximal = mol.GetAtomWithIdx(proximal_id).GetAtomicNum()
    terminal = mol.GetAtomWithIdx(terminal_id).GetAtomicNum()
    return _SITE_TYPE_BY_PAIR.get((proximal, terminal))


def peroxol_seniority(mol, group):
    """P-63.7 ordering for recognized peroxols; preserve other class keys."""
    rank = _SITE_RANK_BY_TYPE.get(peroxol_site_type(mol, group))
    return (18, rank) if rank is not None else group.seniority


def principal_peroxol_sites(mol, groups):
    """Select the lowest-seniority principal site(s), including class 17 ties."""
    groups = tuple(groups)
    if not groups:
        return ()
    first = min(peroxol_seniority(mol, group) for group in groups)
    return tuple(group for group in groups
                 if peroxol_seniority(mol, group) == first)


def suffix_operation(mol, group):
    """Typed suffix operation consumed by chain and ring parent renderers."""
    roles = suffix_roles(mol, group)
    if not roles:
        return None
    # Explicit O/S labels and isotopic H need role-specific isotope grammar.
    chalcogen_ids = (roles.get("proximal", roles.get("proximal_oxygen")),
                     roles.get("terminal", roles.get("terminal_oxygen")))
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in chalcogen_ids):
        return None
    if any(n.GetAtomicNum() == 1 and n.GetIsotope()
           for i in chalcogen_ids for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return None
    site_type = peroxol_site_type(mol, group)
    text = {
        "OOH": "peroxol",
        "SO": "SO-thioperoxol", "SeO": "SeO-selenoperoxol",
        "TeO": "TeO-telluroperoxol", "OS": "OS-thioperoxol",
        "OSe": "OSe-selenoperoxol", "OTe": "OTe-telluroperoxol",
        "SS": "dithioperoxol", "SeS": "SeS-selenothioperoxol",
        "TeS": "TeS-tellurothioperoxol", "SSe": "SSe-selenothioperoxol",
        "STe": "STe-tellurothioperoxol", "SeSe": "diselenoperoxol",
        "TeSe": "TeSe-selenotelluroperoxol",
        "SeTe": "SeTe-selenotelluroperoxol", "TeTe": "ditelluroperoxol",
    }.get(site_type)
    if text is None:
        return None
    return {"text": text, "site_type": site_type, "host": roles["host"],
            "atoms": chalcogen_ids, "roles": roles}


def _has_parent_isotope_locant_requirement(mol, parent_atoms):
    """P-82.6.1.1 guard for partially labeled parent skeletons."""
    atoms = tuple(mol.GetAtomWithIdx(i) for i in parent_atoms)
    carbons = tuple(a for a in atoms if a.GetAtomicNum() == 6)
    carbon_masses = tuple(a.GetIsotope() for a in carbons)
    if any(carbon_masses):
        # P-82.6.1.3 permits omission when all equivalent positions are fully
        # modified in the same way. Keep the exception narrow: every mapped
        # parent carbon must carry the same nonzero isotope.
        uniformly_full = bool(carbons) and len(set(carbon_masses)) == 1 and carbon_masses[0]
        if not uniformly_full:
            return True
    # Isotopic parent hydrogens can require numeric locants under P-82.6.1.1.
    # Do not infer the all-position exception for H from a partial graph view.
    if any(n.GetAtomicNum() == 1 and n.GetIsotope()
           for atom in atoms for n in atom.GetNeighbors()):
        return True
    return False


def _moved_peroxol_smiles(mol, Chem, group, target):
    """Canonical isomeric graph after moving one recognized suffix bond."""
    if target == group.owner:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    edit = Chem.RWMol(mol)
    host_atom = edit.GetAtomWithIdx(group.owner)
    target_atom = edit.GetAtomWithIdx(target)
    if host_atom.GetNoImplicit():
        host_atom.SetNumExplicitHs(host_atom.GetNumExplicitHs() + 1)
    if target != group.owner and target_atom.GetNoImplicit():
        if target_atom.GetNumExplicitHs() < 1:
            return None
        target_atom.SetNumExplicitHs(target_atom.GetNumExplicitHs() - 1)
    edit.RemoveBond(group.owner, group.center)
    edit.AddBond(target, group.center, Chem.BondType.SINGLE)
    moved = edit.GetMol()
    try:
        moved.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(moved)
        return Chem.MolToSmiles(moved, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def peroxol_locant_omission_proven(mol, Chem, groups, parent_atoms, *, allow_ring=False,
                                     force_locants=False):
    """Prove that moving one peroxol suffix within a mapped parent makes no isomer.

    `parent_atoms` is the original-graph atom sequence used by the parent
    writer. The graph comparison preserves substituents, isotopes and
    stereochemistry. Ring omission is opt-in so a caller must first establish
    that its ring numbering route permits a locant-free suffix.
    """
    groups = tuple(groups)
    parent_atoms = tuple(parent_atoms)
    if (force_locants or len(groups) != 1 or len(parent_atoms) < 1
            or len(set(parent_atoms)) != len(parent_atoms)
            or not principal_scope(mol, groups)):
        return False
    if any(i < 0 or i >= mol.GetNumAtoms() for i in parent_atoms):
        return False
    atoms = tuple(mol.GetAtomWithIdx(i) for i in parent_atoms)
    if any(atom.GetAtomicNum() != 6 for atom in atoms):
        return False
    parent_edges = {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                    for bond in iter_bonds(mol)
                    if bond.GetBeginAtomIdx() in parent_atoms
                    and bond.GetEndAtomIdx() in parent_atoms}
    degree = {i: sum(i in edge for edge in parent_edges) for i in parent_atoms}
    if any(value > 2 for value in degree.values()):
        return False
    if len(parent_atoms) > 1 and any(value == 0 for value in degree.values()):
        return False
    in_ring = any(atom.IsInRing() for atom in atoms)
    if in_ring:
        if not allow_ring or len(parent_edges) != len(parent_atoms) or any(v != 2 for v in degree.values()):
            return False
    elif len(parent_edges) != len(parent_atoms) - 1:
        return False
    # The parent atom sequence describes one acyclic chain or one eligible
    # monocycle, not disconnected/fused collections.
    reached = {parent_atoms[0]}
    while True:
        expanded = reached | {j for edge in parent_edges if set(edge) & reached for j in edge}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(parent_atoms):
        return False
    # Explicit hydrogen nodes would need to be transferred during relocation;
    # retain locants until that edit can preserve their graph identity.
    if any(n.GetAtomicNum() == 1 for i in parent_atoms
           for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return False
    # Relinking around a stereo element requires explicit parity correction;
    # leave such cases to locanted naming rather than assuming equivalence.
    if any(int(atom.GetChiralTag()) for atom in iter_atoms(mol)):
        return False
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in iter_bonds(mol)):
        return False
    if _has_parent_isotope_locant_requirement(mol, parent_atoms):
        return False
    group = groups[0]
    if group.owner not in parent_atoms:
        return False
    # P-14.3.4.4 also considers interchange with other prefixes. A pure
    # relocation proof is insufficient when, for example, carbon branches
    # block the other host positions. Keep a locant for decorated multiatom
    # parents until those interchange operations have been proved too.
    if len(parent_atoms) > 1 and any(
        neighbor.GetAtomicNum() > 1
        and neighbor.GetIdx() not in set(parent_atoms) | set(group.atoms)
        for atom in atoms for neighbor in atom.GetNeighbors()
    ):
        return False
    original = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # Every other parent carbon bearing an ordinary hydrogen is a possible
    # relocation site. If all resulting graphs are identical, no isomer is
    # produced by moving the suffix within this selected parent.
    for target in parent_atoms:
        atom = mol.GetAtomWithIdx(target)
        if target != group.owner and atom.GetTotalNumHs(includeNeighbors=True) < 1:
            continue
        moved = _moved_peroxol_smiles(mol, Chem, group, target)
        if moved is None or moved != original:
            return False
    return True


def peroxol_ending(mol, Chem, parent, groups, locants, *, parent_atoms=(),
                    allow_ring_omission=False, force_locants=False):
    """Render a mapped one-or-more peroxol suffix on a complete parent."""
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings
    from .formatting import attach_suffix, multiplied_suffix

    if not groups or not principal_scope(mol, groups) or len(groups) != len(locants):
        return "", frozenset()
    operations = [suffix_operation(mol, group) for group in groups]
    if any(operation is None for operation in operations):
        return "", frozenset()
    site_types = {operation["site_type"] for operation in operations}
    if len(site_types) != 1:
        return "", frozenset()
    omit_single_locant = (
        len(groups) == 1 and len(locants) == 1
        and peroxol_locant_omission_proven(
            mol, Chem, groups, parent_atoms, allow_ring=allow_ring_omission,
            force_locants=force_locants,
        )
    )
    site_type = next(iter(site_types))
    if site_type == "OOH":
        suffix = multiplied_suffix(len(groups), "peroxol")
        if not suffix:
            return "", frozenset()
        text = attach_suffix(parent, suffix, () if omit_single_locant else locants)
    else:
        # P-16.3.3 multiplies functional/cumulative suffixes uniformly; the
        # P-63.4.2 Table 6.1 suffix is selected only when all principal sites
        # have the same atom-order type. P-16.7 boundary elision is applied by
        # attach_suffix for dithioperoxol. SO/OS retain Table 6.1's internal
        # hyphen following their atom-order prefix.
        suffix_base = operations[0]["text"]
        suffix = multiplied_suffix(len(groups), suffix_base) if suffix_base else ""
        if not suffix:
            return "", frozenset()
        if site_type in {"SO", "OS", "SeO", "TeO", "OSe", "OTe",
                         "SeS", "TeS", "SSe", "STe", "TeSe", "SeTe"}:
            displayed = () if omit_single_locant else locants
            text = (f"{parent}-{','.join(map(str, displayed))}-{suffix}"
                    if displayed else f"{parent}-{suffix}")
        else:
            text = attach_suffix(parent, suffix, () if omit_single_locant else locants)
    # The two chalcogen atoms belong to the typed suffix unit and bind to the
    # host locant for stereo/isotope accounting. Their isotopes were declined.
    mapping = {}
    for operation, locant in zip(operations, locants):
        roles = operation["roles"]
        proximal = roles.get("proximal", roles.get("proximal_oxygen"))
        terminal = roles.get("terminal", roles.get("terminal_oxygen"))
        if proximal is None or terminal is None:
            return "", frozenset()
        mapping[proximal] = str(locant)
        mapping[terminal] = str(locant)
    bindings = hydrogen_bindings(mol, mapping)
    component = MappedComponent(text, tuple(bindings.items()), frozenset(bindings),
                                local_isotope_atoms=frozenset(bindings))
    # The oxygen unit's isotopes were rejected; local ownership prevents the
    # suffix oxygen roles from becoming carbon-parent isotope locants.
    rendered = decorate_component(mol, Chem, component)
    return (rendered, frozenset(bindings)) if rendered else ("", frozenset())


def hydroperoxy_prefix(mol, Chem, root, blocked, atoms):
    """Map a C–O–O–H substituent as the P-63.4.1 `hydroperoxy` prefix."""
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings

    # `root` is the proximal oxygen directly attached to the higher-priority
    # parent and `blocked` is its carbon host.
    group = next((g for g in recognize_peroxols(mol)
                  if g.owner == blocked and g.center == root
                  and peroxol_site_type(mol, g) == "OOH"), None)
    if group is None:
        return "", frozenset()
    roles = suffix_roles(mol, group)
    if not roles:
        return "", frozenset()
    owned = {roles["proximal_oxygen"], roles["terminal_oxygen"]}
    owned.update(n.GetIdx() for i in tuple(owned)
                 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetAtomicNum() == 1)
    if not set(atoms) <= owned:
        return "", frozenset()
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms):
        return "", frozenset()
    if any(n.GetAtomicNum() == 1 and n.GetIsotope()
           for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return "", frozenset()
    mapping = hydrogen_bindings(mol, {roles["proximal_oxygen"]: "", roles["terminal_oxygen"]: ""})
    component = MappedComponent("hydroperoxy", tuple(mapping.items()), frozenset(mapping),
                                isotope_unlocanted_elements=())
    rendered = decorate_component(mol, Chem, component)
    return (rendered, frozenset(owned | set(mapping))) if rendered else ("", frozenset())


def chalcogen_peroxol_prefix(mol, Chem, root, blocked, atoms):
    """Render a non-OOH Table 6.1 group as its P-63.4.2.2 prefix.

    The bridge root is the atom attached directly to the higher-priority
    parent; `blocked` is the carbon on the other side of that atom.
    """
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings

    group = next((g for g in recognize_peroxols(mol)
                  if g.owner == blocked and g.center == root
                  and peroxol_site_type(mol, g) not in {None, "OOH"}), None)
    if group is None:
        return "", frozenset()
    roles = suffix_roles(mol, group)
    if not roles:
        return "", frozenset()
    proximal_id = roles.get("proximal", roles.get("proximal_oxygen"))
    terminal_id = roles.get("terminal", roles.get("terminal_oxygen"))
    proximal = mol.GetAtomWithIdx(proximal_id)
    terminal = mol.GetAtomWithIdx(terminal_id)
    # Isotopic O/S locants and isotopic hydrogen need dedicated prefix
    # rendering. Keep their identities intact by declining these structures.
    if any(atom.GetIsotope() for atom in (proximal, terminal)):
        return "", frozenset()
    chalcogen_ids = {proximal_id, terminal_id}
    explicit_h = {neighbor.GetIdx() for idx in chalcogen_ids
                  for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                  if neighbor.GetAtomicNum() == 1}
    if any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in explicit_h):
        return "", frozenset()
    owned = chalcogen_ids | explicit_h
    if not set(atoms) <= owned:
        return "", frozenset()
    names = {
        "SO": "hydroxysulfanyl", "SeO": "hydroxyselanyl",
        "TeO": "hydroxytellanyl", "OS": "sulfanyloxy",
        "OSe": "selanyloxy", "OTe": "tellanyloxy",
        "SS": "disulfanyl", "SeS": "sulfanylselanyl",
        "TeS": "sulfanyltellanyl", "SSe": "selanylsulfanyl",
        "STe": "tellanylsulfanyl", "SeSe": "diselanyl",
        "TeSe": "selanyltellanyl", "SeTe": "tellanylselanyl",
        "TeTe": "ditellanyl",
    }
    name = names[peroxol_site_type(mol, group)]
    mapping = hydrogen_bindings(mol, {idx: "" for idx in chalcogen_ids})
    component = MappedComponent(name, tuple(mapping.items()), frozenset(mapping),
                                isotope_unlocanted_elements=())
    rendered = decorate_component(mol, Chem, component)
    return (rendered, frozenset(owned | set(mapping))) if rendered else ("", frozenset())


def chain_proposals(mol, Chem, inventory, *, scope=None, principal=None):
    """Generate mapped acyclic carbon-chain proposals carrying peroxol suffixes."""
    from .multifunctional_parent import _paths
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
    from .formatting import complete_parent_name
    from .branch_rules import parent_chain_name
    from .parent_selection import graph_parent_facts
    from .cross_route_selection import ParentProposal, _annotation_ownership

    groups = inventory.principal if principal is None else tuple(principal)
    if (not groups or groups[0].kind != "peroxol"
            or not principal_scope(mol, groups)):
        return
    carbon = {a.GetIdx() for a in iter_atoms(mol)
              if a.GetAtomicNum() == 6 and not a.IsInRing()
              and (scope is None or a.GetIdx() in scope)}
    for path in _paths(mol, carbon):
        for seq in (path, tuple(reversed(path))):
            covered = tuple(sorted((g for g in groups if g.owner in seq),
                                   key=lambda g: seq.index(g.owner)))
            if not covered or len(covered) != len(groups):
                continue
            carbon_mapping = {idx: str(pos) for pos, idx in enumerate(seq, 1)}
            peroxol_locants = tuple(sorted((int(carbon_mapping[g.owner]) for g in covered)))
            oxygen_mapping = {}
            failed = False
            prefixes, embedded = [], set()
            for group in covered:
                operation = suffix_operation(mol, group)
                if operation is None:
                    failed = True
                    break
                locant = carbon_mapping[group.owner]
                role_map = operation["roles"]
                proximal = role_map.get("proximal", role_map.get("proximal_oxygen"))
                terminal = role_map.get("terminal", role_map.get("terminal_oxygen"))
                oxygen_mapping[proximal] = locant
                oxygen_mapping[terminal] = locant
            if failed:
                continue
            mapping = dict(carbon_mapping)
            mapping.update(oxygen_mapping)
            for idx in seq:
                for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root = atom.GetIdx()
                    if root in mapping or atom.GetAtomicNum() <= 1 or (scope is not None and root not in scope):
                        continue
                    text, owned = mapped_branch(mol, Chem, root, idx)
                    if not text or owned & (mapping.keys() | embedded) or any(c.isspace() for c in text):
                        failed = True
                        break
                    prefixes.append((carbon_mapping[idx], text))
                    embedded.update(owned)
                if failed:
                    break
            mapping = hydrogen_bindings(mol, mapping)
            needed = {a.GetIdx() for a in iter_atoms(mol)
                      if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())}
            target = needed if scope is None else needed & set(scope)
            if failed or set(mapping) | embedded != target or not _annotation_ownership(mol, mapping):
                continue
            parent = complete_parent_name(parent_chain_name(mol, seq))
            if not parent:
                continue
            ending, suffix_owned = peroxol_ending(mol, Chem, parent, covered,
                                                   tuple(map(str, peroxol_locants)),
                                                   parent_atoms=seq)
            if not ending:
                continue
            name = decorate_component(mol, Chem, MappedComponent(
                ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
                frozenset(embedded), systematic_parent=True,
                local_isotope_atoms=suffix_owned,
            ))
            if not name or len(name) > 260:
                continue
            facts = graph_parent_facts(
                mol, seq, mapping, principal_class=groups[0].seniority,
                suffix_centers=[g.center for g in covered], suffix_locants=peroxol_locants,
                prefixes=prefixes, kind="chain", name=name, allow_annotations=True,
            )
            if facts is not None:
                yield ParentProposal(
                    name, "mapped-chain-peroxol", "peroxol",
                    frozenset(g.center for g in covered), tuple(seq),
                    tuple(mapping.items()), facts, True,
                )
