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
    if (n.IsInRing() or int(n.GetChiralTag()) or n.GetNumRadicalElectrons()
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
    if any(a.GetAtomicNum() not in {1, 6, 7, 8} for a in mol.GetAtoms()):
        return None
    # Hydrogen isotopes and N labels need separate parent/suffix placement
    # proof. Carbon and nonoxide oxygen labels are handled by mapped components.
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) > 1 or any(
        len(ring) != 6 or any(
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring
        ) for ring in rings
    ):
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
    allowed = {"amine", "alcohol", "ketone", "acid", "amide"}
    if any(group.kind not in allowed for group in inventory.groups):
        return None
    amines = [g for g in inventory.groups if g.kind == "amine"]
    if (not any(g.center == oxide_n for g in amines)
            or len(amines) > 2
            or any(not ordinary_amine(private, g.center) for g in amines)):
        return None
    # Every nonoxide heteroatom must be assigned to one of the recognized
    # groups. This excludes untyped ethers, peroxides, nitro groups and salts.
    covered = {idx for group in inventory.groups for idx in group.atoms}
    for atom in private.GetAtoms():
        if atom.GetAtomicNum() in {7, 8} and atom.GetIdx() not in covered:
            return None
    counts = {}
    for group in inventory.groups:
        counts[group.kind] = counts.get(group.kind, 0) + 1
    if any(count > 1 for kind, count in counts.items() if kind != "amine"):
        return None
    return inventory, tuple(amines)


def functional_amine_oxide_decision(mol, Chem):
    """Construct a mapped amine-oxide name, or decline outside proved scope.

    Admitted graphs contain one neutral tertiary N+–O− oxide, one or more
    subordinate P-62.5 prefix contexts from the admitted vocabulary, and a
    chain-based ordinary amine parent. A single aromatic carbon ring is
    permitted as a mapped branch. Other amines, multiple characteristic sites
    of one kind, salt contexts, and unsupported parent constructions decline.
    """
    site = _oxide_site(mol, Chem)
    if site is None:
        return None
    nitrogen, oxygen = site
    private = _neutral_copy(mol, Chem, nitrogen, oxygen)
    if private is None:
        return None
    private_n = next(a.GetIdx() for a in private.GetAtoms()
                     if a.GetAtomMapNum() == nitrogen + 1)
    context = _admitted_context(private, private_n)
    if context is None:
        return None
    inventory, amines = context
    oxide_amine = next((g for g in amines if g.center == private_n), None)
    if oxide_amine is None:
        return None
    # The added route owns the additional-group constructions; the older
    # single-group route keeps its original narrow contract.
    if len(inventory.groups) <= 1:
        return None

    from .annotation_transfer import mapped_stereochemistry_equivalent
    if mapped_stereochemistry_equivalent(mol, private, Chem) is not True:
        return None

    from .amine_components import chain_proposals
    from .parent_selection import parent_frontier

    # In a P-62.5 functional-class name the oxide is senior to these other
    # characteristic groups. Ask the amine parent constructor to enumerate
    # complete mapped amine candidates directly, rather than letting the
    # ordinary inventory choose an acid/amide/ketone suffix first.
    forced = SimpleNamespace(principal=(oxide_amine,))
    proposals = list(chain_proposals(
        private, Chem, forced, principal=(oxide_amine,)
    ))
    if not proposals:
        return None
    # P-65 carboxy and carbamoyl prefixes keep their carbonyl centers outside
    # the parent skeleton. The typed branch renderer retains their atoms.
    excluded_centers = {
        g.center for g in inventory.groups if g.kind in {"acid", "amide"}
    }
    proposals = [p for p in proposals
                 if not excluded_centers.intersection(p.parent_atoms)]
    if not proposals:
        return None
    frontier = parent_frontier(proposals, lambda p: p.parent_selection)
    names = {p.name for p in frontier if p.annotation_coverage and p.name}
    if len(names) != 1:
        return None
    name = next(iter(names))
    # P-65.2.1.5 and the P-62.5 example use carbamoyl for an amide-derived
    # branch. Older general branch renderers may emit the traditional -yl
    # spelling, so normalize only the typed amide prefix token in this route.
    if any(g.kind == "amide" for g in inventory.groups):
        name = name.replace("carbamyl", "carbamoyl")
    oxide_atom = mol.GetAtomWithIdx(oxygen)
    oxide_word = (f"N-({oxide_atom.GetIsotope()}O)oxide"
                  if oxide_atom.GetIsotope() else "N-oxide")
    name += " " + oxide_word
    candidates = [p for p in frontier if p.name in names and p.annotation_coverage]
    if not candidates:
        return None
    proposal = candidates[0]

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
        atom_locants=tuple(atom_locants) + ((oxygen, "N"),),
        heterounits=(),
        component_roles=tuple(roles),
    )
