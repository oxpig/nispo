"""Mapped functional-class construction for simple tertiary amine oxides.

P-62.5 method (1) names one amine oxide from its preferred amine name followed
by N-oxide.  Parent selection is delegated to the existing ordinary amine
route on a private copy with only the N-oxide oxygen removed.  All returned
atom identities and annotations are remapped to the original zwitterion.
"""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


RULE = "P-62.5(1)/P-74.2.1.2/P-41/P-62.2.2.1/P-44/P-45/P-82/P-9"


def _site(mol, Chem):
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1]
    if len(nitrogens) != 1 or len(oxygens) != 1:
        return None
    n, o = nitrogens[0], oxygens[0]
    if (n.IsInRing() or int(n.GetChiralTag()) or n.GetNumRadicalElectrons()
            or n.GetIsAromatic() or n.GetTotalValence() != 4
            or o.GetIsAromatic() or o.GetNumRadicalElectrons()
            or o.GetDegree() != 1 or o.GetTotalValence() != 1):
        return None
    bond = mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    carbon = []
    for atom in n.GetNeighbors():
        if atom.GetIdx() == o.GetIdx():
            continue
        bond = mol.GetBondBetweenAtoms(n.GetIdx(), atom.GetIdx())
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or bond is None or bond.GetBondType() != Chem.BondType.SINGLE):
            return None
        carbon.append(atom.GetIdx())
    if len(carbon) != 3 or n.GetTotalNumHs(includeNeighbors=True) != 0:
        return None
    if any((a.GetFormalCharge() and a.GetIdx() not in {n.GetIdx(), o.GetIdx()})
           or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() not in {1, 6, 7, 8} for a in mol.GetAtoms()):
        return None
    # This route is deliberately restricted to carbon parent/branch families.
    # Heteroatoms other than the unique N-oxide unit require their own class
    # construction and class-seniority comparison.
    if any(a.GetAtomicNum() not in {1, 6} for a in mol.GetAtoms()
           if a.GetIdx() not in {n.GetIdx(), o.GetIdx()}):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) > 1 or any(
        mol.GetAtomWithIdx(i).GetAtomicNum() != 6
        for ring in rings for i in ring
    ):
        return None
    return n.GetIdx(), o.GetIdx(), tuple(carbon)


def _neutral_amine_copy(mol, Chem, nitrogen, oxygen):
    """Remove O- from a mapped private copy and neutralize its tertiary N."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    rw.RemoveAtom(oxygen)
    neutral = rw.GetMol()
    nidx = next(a.GetIdx() for a in neutral.GetAtoms()
                if a.GetAtomMapNum() == nitrogen + 1)
    n = neutral.GetAtomWithIdx(nidx)
    n.SetFormalCharge(0)
    try:
        Chem.SanitizeMol(neutral)
    except Exception:
        return None
    return neutral


def amine_oxide_decision(mol, Chem, *, allow_unresolved_parent=False):
    """Return a complete `MethodDecision`, or `None` outside the proved scope.

    Admitted graphs have one neutral zwitterionic N+–O− site, tertiary N bound
    by single bonds to three carbon groups, and no other heteroatom or charge.
    The ordinary amine route must produce a complete mapped parent on the
    private neutral copy. Carbon stereochemistry and carbon/N isotopes are
    retained by that mapped route; an N stereotag declines because neutralizing
    the private parent would change the four-ligand P-93.2.3 comparison.
    """
    site = _site(mol, Chem)
    if site is None:
        return None
    nitrogen, oxygen, _ = site
    if allow_unresolved_parent:
        return _zwitterionic_oxide(mol, Chem, site)
    private = _neutral_amine_copy(mol, Chem, nitrogen, oxygen)
    if private is None:
        return None
    from .component_mapping import chemical_inventory
    inventory = chemical_inventory(private)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'amine'
            or inventory.principal[0].center != next(
                a.GetIdx() for a in private.GetAtoms()
                if a.GetAtomMapNum() == nitrogen + 1)):
        return None
    from .annotation_transfer import mapped_stereochemistry_equivalent
    if mapped_stereochemistry_equivalent(mol, private, Chem) is not True:
        return None
    # A private parent proposal is used only if the established graph route
    # can construct it and all mapped atoms survive as original roles.
    from .cross_route_selection import cross_route_selection_outcome
    outcome = cross_route_selection_outcome(private, Chem)
    proposal = (outcome.selected if allow_unresolved_parent else
                outcome.selection.selected if outcome.selection is not None else None)
    if proposal is None or not proposal.name:
        return None
    if (not proposal.annotation_coverage or proposal.principal_kind != 'amine'
            or proposal.principal_centers != frozenset({
                next(a.GetIdx() for a in private.GetAtoms()
                     if a.GetAtomMapNum() == nitrogen + 1)
            })):
        return None
    # The full decision graph includes explicit isotopic H atoms when present.
    original_atoms = frozenset(a.GetIdx() for a in mol.GetAtoms()
                               if a.GetAtomicNum() > 1)
    private_to_original = {
        a.GetIdx(): a.GetAtomMapNum() - 1 for a in private.GetAtoms()
        if a.GetAtomicNum() > 1
    }
    if set(private_to_original.values()) | {oxygen} != set(
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1
    ):
        return None
    atom_locants = []
    for idx, locant in proposal.atom_locants:
        if idx not in private_to_original:
            return None
        atom_locants.append((private_to_original[idx], locant))
    if len({idx for idx, _ in atom_locants}) != len(atom_locants):
        return None
    # In this two-word construction, N is the locant for the oxide operation;
    # the oxygen isotope descriptor follows that locant and precedes "oxide"
    # (P-82.2.4/P-82.6.2). This placement is rule-derived, not an OPSIN repair.
    oxygen_atom = mol.GetAtomWithIdx(oxygen)
    oxide_word = 'N-oxide'
    if oxygen_atom.GetIsotope():
        oxide_word = f"N-({oxygen_atom.GetIsotope()}O)oxide"
    name = proposal.name + ' ' + oxide_word
    core = {private_to_original[i] for i in proposal.parent_atoms
            if i in private_to_original}
    core.add(nitrogen)
    core_locants = dict(atom_locants)
    if not core <= set(core_locants) or oxygen in core:
        return None
    role_rows = [(
        'amine-parent', frozenset(core),
        tuple(sorted((i, core_locants[i]) for i in core)),
    )]
    remaining = set(original_atoms) - core - {oxygen}
    while remaining:
        start = min(remaining)
        component = set()
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in component or idx not in remaining:
                continue
            component.add(idx)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if n.GetIdx() in remaining and n.GetAtomicNum() > 1)
        remaining -= component
        attachments = {n.GetIdx() for idx in component
                       for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                       if n.GetIdx() in core}
        if len(attachments) != 1:
            return None
        attachment = next(iter(attachments))
        locant = core_locants.get(attachment)
        if locant is None:
            return None
        role_rows.append((
            'amine-substituent', frozenset(component),
            # The branch has its own internal component numbering.  The
            # attachment locant belongs to the host amine parent, so do not
            # claim it as a locant for every atom in this disjoint role.
            (),
        ))
    role_rows.append(('N-oxide-oxygen', frozenset({oxygen}), ()))
    roles = tuple(role_rows)
    if {idx for _, role_atoms, _ in roles for idx in role_atoms} != set(original_atoms):
        return None
    return MethodDecision(
        name=name,
        method='functional-class',
        principal_centers=frozenset({nitrogen}),
        atoms=original_atoms,
        rule=RULE,
        atom_locants=tuple(atom_locants) + ((oxygen, 'N'),),
        heterounits=(),
        component_roles=roles,
        preference_issues=(() if outcome.status == 'complete' else
            (SelectionLimitation(
                'amine-oxide-parent', 'P-62.5/P-44/P-45', outcome.reason),)),
    )


def _zwitterionic_oxide(mol, Chem, site):
    """P-74.2.1.2(1): mapped cationic substituent on an oxidanide parent.

    This is a general construction, never a preferred functional-class claim.
    Each carbon side keeps its own mapped prefix. The N and O nuclides belong
    to their actual ionic components, including the original N--O connection.
    """
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .formatting import enclose_name_unit

    nitrogen, oxygen, roots = site
    embedded, prefixes = set(), []
    for root in roots:
        text, owned = mapped_branch(mol, Chem, root, nitrogen)
        if not text or embedded & owned or {nitrogen, oxygen} & owned:
            return None
        prefixes.append(('', text))
        embedded.update(owned)
    required = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if embedded | {nitrogen, oxygen} != required:
        return None
    n_component = MappedComponent(
        'azaniumyl', ((nitrogen, ''),), frozenset({nitrogen}),
        tuple(prefixes), frozenset(embedded), prefix_mode='unlocanted')
    cation = decorate_component(mol, Chem, n_component)
    anion = decorate_component(mol, Chem, MappedComponent(
        'oxidanide', ((oxygen, ''),), frozenset({oxygen})))
    if not cation or not anion:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(
        name=enclose_name_unit(cation) + anion, method='substitutive',
        principal_centers=frozenset({oxygen}), atoms=heavy,
        rule='P-74.2.1.2(1)/P-82.2.1/P-29',
        component_roles=(('cationic-substituent', heavy - {oxygen}, ()),
                         ('oxidanide-parent', frozenset({oxygen}), ((oxygen, ''),))),
        preference_issues=(SelectionLimitation('amine-oxide-method', 'P-74.2.1.2',
            'The mapped zwitterion is a general name; functional-class nomenclature is preferred.'),))

