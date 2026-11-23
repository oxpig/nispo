"""Mapped P-62.5 amine oxides with subordinate characteristic groups.

This is a separate, deliberately bounded producer from the simple
``amine_oxide_decision`` route. It selects a complete ordinary amine parent
with mapped chain proposals, while P-62.5 makes the single amine oxide the
principal class and subordinate groups become prefixes.
"""

from types import SimpleNamespace

from .naming_methods import MethodDecision


RULE = "P-62.5(1)/P-41/P-44/P-45/P-62.2/P-63/P-65/P-82/P-9"


def _oxide_site(mol, Chem):
    # Keep the exploratory chain search within the same bounded graph size as
    # the amine-parent route that consumes it.
    if mol.GetNumHeavyAtoms() > 64:
        return None
    nitrogens = [a for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1]
    oxygens = [a for a in mol.GetAtoms()
               if a.GetAtomicNum() == 8 and a.GetFormalCharge() == -1]
    if len(nitrogens) != 1 or len(oxygens) != 1:
        return None
    n, o = nitrogens[0], oxygens[0]
    if (int(n.GetChiralTag()) or n.GetNumRadicalElectrons()
            or n.GetIsAromatic() or n.GetTotalValence() != 4
            or n.GetIsotope() or o.GetIsAromatic() or o.GetNumRadicalElectrons()
            or o.GetDegree() != 1 or o.GetTotalValence() != 1):
        return None
    bond = mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    carbon_neighbors = []
    for atom in n.GetNeighbors():
        if atom.GetIdx() == o.GetIdx():
            continue
        bond = mol.GetBondBetweenAtoms(n.GetIdx(), atom.GetIdx())
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or
                bond is None or bond.GetBondType() != Chem.BondType.SINGLE):
            return None
        carbon_neighbors.append(atom.GetIdx())
    if len(carbon_neighbors) != 3 or n.GetTotalNumHs(includeNeighbors=True) != 0:
        return None
    if any(a.GetNumRadicalElectrons() or
           (a.GetFormalCharge() and a.GetIdx() not in {n.GetIdx(), o.GetIdx()})
           for a in mol.GetAtoms()):
        return None
    # Hydrogen isotopes and N labels need separate parent/suffix placement
    # proof. Carbon and nonoxide oxygen labels are handled by mapped components.
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None
    return n.GetIdx(), o.GetIdx()


def _neutral_copy(mol, Chem, nitrogen, oxygen):
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    rw.RemoveAtom(oxygen)
    private = rw.GetMol()
    nidx = next(a.GetIdx() for a in private.GetAtoms()
                if a.GetAtomMapNum() == nitrogen + 1)
    private.GetAtomWithIdx(nidx).SetFormalCharge(0)
    try:
        Chem.SanitizeMol(private)
    except Exception:
        return None
    return private


def _admitted_context(private, oxide_n):
    """Return recognized groups only for the source-backed prefix vocabulary."""
    from .extended_inventory import selection_inventory
    from .amine_components import ordinary_amine

    inventory = selection_inventory(private)
    if not inventory.supported:
        return None
    amines = [g for g in inventory.groups if g.kind == "amine"]
    if (not any(g.center == oxide_n for g in amines)
            or len(amines) > 2
            or any(not ordinary_amine(private, g.center) for g in amines)):
        return None
    return inventory, tuple(amines)


def _mapped_oxide_parent(private, Chem, private_n):
    from .mapped_ring_parents import parent_facts as ring_facts, records, ring_systems
    from .parent_selection import parent_frontier

    n = private.GetAtomWithIdx(private_n)
    if n.IsInRing():
        core = next((core for core in ring_systems(private) if private_n in core), ())
        if (len(core) not in {5, 6, 7} or any(private.GetRingInfo().NumAtomRings(i) != 1 for i in core)
                or any(a.GetAtomicNum() in {7, 8, 16} and b.GetBondTypeAsDouble() > 1
                       for child in n.GetNeighbors() for b in child.GetBonds() for a in (b.GetOtherAtom(child),))):
            return None
        rows = records(private, Chem, core_filter=core, principal=(), general_branches=True)
        if len(rows) == 1 and rows[0].annotation_coverage and rows[0].hydrogen_budget_proved:
            return rows[0], dict(rows[0].atom_locants)[private_n]
        return None

    context = _admitted_context(private, private_n)
    if context is None:
        return None
    inventory, amines = context
    oxide_amine = next((g for g in amines if g.center == private_n), None)
    if oxide_amine is None:
        return None
    from .amine_components import chain_proposals

    # In a P-62.5 functional-class name the oxide is senior to these other
    # characteristic groups. Ask the amine parent constructor to enumerate
    # complete mapped amine candidates directly, rather than letting the
    # ordinary inventory choose an acid/amide/ketone suffix first.
    forced = SimpleNamespace(principal=(oxide_amine,))
    proposals = [(row, row.parent_selection) for row in chain_proposals(
        private, Chem, forced, principal=(oxide_amine,))]
    hosts = {a.GetIdx() for a in n.GetNeighbors() if a.GetIsAromatic()}
    for core in ring_systems(private):
        if len(hosts & core) == 1:
            proposals.extend((row, ring_facts(private, row, oxide_amine.seniority)) for row in
                             records(private, Chem, core_filter=core, principal=(oxide_amine,),
                                     general_branches=True) if row.hydrogen_budget_proved)
    # P-65 carboxy and carbamoyl prefixes keep their carbonyl centers outside
    # the parent skeleton. The typed branch renderer retains their atoms.
    excluded_centers = {
        g.center for g in inventory.groups if g.kind in {"acid", "amide"}
    }
    proposals = [(row, facts) for row, facts in proposals if facts is not None
                 and row.annotation_coverage and row.name
                 and not excluded_centers.intersection(row.parent_atoms)]
    if not proposals:
        return None
    frontier = parent_frontier(proposals, lambda option: option[1])
    if len({row.name for row, _ in frontier}) != 1:
        return None
    return frontier[0][0], 'N'


def functional_amine_oxide_decision(mol, Chem):
    """Map the preferred functional-class name of one tertiary amine oxide."""
    site = _oxide_site(mol, Chem)
    if site is None:
        return None
    nitrogen, oxygen = site
    private = _neutral_copy(mol, Chem, nitrogen, oxygen)
    if private is None:
        return None
    private_n = next(a.GetIdx() for a in private.GetAtoms() if a.GetAtomMapNum() == nitrogen + 1)
    from .annotation_transfer import mapped_stereochemistry_equivalent

    if mapped_stereochemistry_equivalent(mol, private, Chem) is not True:
        return None
    choice = _mapped_oxide_parent(private, Chem, private_n)
    if choice is None:
        return None
    proposal, oxide_locant = choice
    # P-65.2.1.5 and the P-62.5 example use carbamoyl for an amide-derived
    # branch. Older general branch renderers may emit the traditional -yl
    # spelling, so normalize only the typed amide prefix token in this route.
    name = proposal.name.replace("carbamyl", "carbamoyl")
    oxide_atom = mol.GetAtomWithIdx(oxygen)
    oxide_word = (f"{oxide_locant}-({oxide_atom.GetIsotope()}O)oxide"
                  if oxide_atom.GetIsotope() else f"{oxide_locant}-oxide")
    name += " " + oxide_word

    private_to_original = {
        a.GetIdx(): a.GetAtomMapNum() - 1 for a in private.GetAtoms()
        if a.GetAtomicNum() > 1
    }
    original_heavy = frozenset(a.GetIdx() for a in mol.GetAtoms()
                               if a.GetAtomicNum() > 1)
    if set(private_to_original.values()) | {oxygen} != set(original_heavy):
        return None
    atom_locants = []
    for idx, locant in proposal.atom_locants:
        if idx not in private_to_original:
            return None
        atom_locants.append((private_to_original[idx], locant))
    if len({idx for idx, _ in atom_locants}) != len(atom_locants):
        return None
    core = {private_to_original[i] for i in proposal.parent_atoms
            if i in private_to_original}
    core.add(nitrogen)
    if oxygen in core:
        return None
    core_locants = dict(atom_locants)
    if not core <= set(core_locants):
        return None
    roles = [(
        "amine-oxide-parent", frozenset(core),
        tuple(sorted((i, core_locants[i]) for i in core)),
    )]
    remaining = set(original_heavy) - core - {oxygen}
    while remaining:
        start = min(remaining)
        component = set()
        stack = [start]
        while stack:
            idx = stack.pop()
            if idx in component or idx not in remaining:
                continue
            component.add(idx)
            stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if a.GetIdx() in remaining and a.GetAtomicNum() > 1)
        remaining -= component
        attachments = {a.GetIdx() for idx in component
                       for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                       if a.GetIdx() in core and a.GetAtomicNum() > 1}
        if len(attachments) != 1:
            return None
        roles.append(("amine-oxide-branch", frozenset(component), ()))
    roles.append(("N-oxide-oxygen", frozenset({oxygen}), ()))
    if {idx for _, atoms, _ in roles for idx in atoms} != set(original_heavy):
        return None
    return MethodDecision(
        name=name,
        method="functional-class",
        principal_centers=frozenset({nitrogen}),
        atoms=original_heavy,
        rule=RULE,
        atom_locants=tuple(atom_locants) + ((oxygen, oxide_locant),),
        heterounits=(),
        component_roles=tuple(roles),
    )
