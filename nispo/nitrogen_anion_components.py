"""Mapped amine/imine hydron-loss operations under P-72.2.2.2.3."""

from dataclasses import dataclass


_RULE = "P-72.2.2.2.3/P-62.2.1.2/P-62.2.2.1/P-62.2.2.2/P-44/P-14"
_MAX_HEAVY_ATOMS = 64


@dataclass(frozen=True, slots=True)
class NitrogenAnionComponent:
    """A complete component with charge owned by original nitrogen atoms."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    component_roles: tuple[tuple[str, frozenset[int], tuple[tuple[int, str], ...]], ...]
    parent_atoms: frozenset[int]
    anionic_center: int
    equivalent_parent_mappings: tuple[tuple[int, ...], ...] = ()
    anionic_centers: tuple[int, ...] = ()
    suffix: str = 'aminide'
    preference_issues: tuple = ()


def requires_nitrogen_anion_proof(mol):
    """Do not send multiply charged carbon-bound N to monoanion fallbacks.

    This deliberately includes characteristic groups outside the producer's
    scope: declining a parent proof cannot authorize losing a hydron/charge.
    It also works on disconnected inputs before a generic salt assembly.
    """
    return any(a.GetAtomicNum() == 7 and a.GetFormalCharge() < -1
               and any(n.GetAtomicNum() == 6 for n in a.GetNeighbors())
               for a in mol.GetAtoms())


def nitrogen_aminide_component(mol, Chem):
    """Return a typed P-72 aminide component for a bounded neutral-amine H loss."""
    if (mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()):
        return None
    atoms = tuple(mol.GetAtoms())
    nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
    if (len(nitrogens) != 1 or any(atom.GetAtomicNum() not in {6, 7} for atom in atoms)
            or any(atom.GetIsotope() or int(atom.GetChiralTag())
                   or atom.GetNumRadicalElectrons() for atom in atoms)
            or any(int(bond.GetStereo()) or bond.GetBondDir() != Chem.BondDir.NONE
                   or bond.GetBondType() != Chem.BondType.SINGLE for bond in mol.GetBonds())):
        return None
    nitrogen = nitrogens[0]
    if (nitrogen.GetFormalCharge() != -1
            or any(atom.GetFormalCharge() for atom in atoms if atom.GetIdx() != nitrogen.GetIdx())):
        return None
    if any(atom.IsInRing() for atom in atoms):
        return None

    heavy = frozenset(atom.GetIdx() for atom in atoms)
    adjacency = {idx: set() for idx in heavy}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left].add(right)
        adjacency[right].add(left)
    if sum(map(len, adjacency.values())) != 2 * (len(heavy) - 1):
        return None
    visited, stack = set(), [nitrogen.GetIdx()]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current] - visited)
    if visited != heavy:
        return None

    nidx = nitrogen.GetIdx()
    arm_starts = tuple(sorted(adjacency[nidx]))
    if len(arm_starts) not in {1, 2} or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                             for i in arm_starts):
        return None
    amine_kind = "primary" if len(arm_starts) == 1 else "secondary"
    expected_n_h = 1 if amine_kind == "primary" else 0
    if nitrogen.GetTotalNumHs(includeNeighbors=True) != expected_n_h:
        return None
    if nitrogen.GetTotalValence() != len(arm_starts) + expected_n_h:
        return None

    arms = []
    owned_carbon = set()
    for start in arm_starts:
        path, previous, current = [start], nidx, start
        while True:
            if current in owned_carbon:
                return None
            owned_carbon.add(current)
            neighbors = adjacency[current] - {previous}
            if not neighbors:
                break
            if len(neighbors) != 1:
                return None
            next_idx = next(iter(neighbors))
            if mol.GetAtomWithIdx(next_idx).GetAtomicNum() != 6:
                return None
            previous, current = current, next_idx
            path.append(current)
            if len(path) > 60:
                return None
        arms.append(tuple(path))
    if owned_carbon != set(heavy) - {nidx}:
        return None
    if any(len(adjacency[idx]) > 2 for idx in owned_carbon):
        return None
    for atom in atoms:
        if atom.GetAtomicNum() == 6:
            degree = len(adjacency[atom.GetIdx()])
            expected_h = 4 - degree
            if (expected_h < 0 or atom.GetTotalNumHs(includeNeighbors=True) != expected_h
                    or atom.GetTotalValence() != degree + expected_h):
                return None

    # Reconstitute the corresponding neutral amine only for parent comparison;
    # output naming and charge ownership remain based on the original graph.
    neutral = Chem.Mol(mol)
    neutral_n = neutral.GetAtomWithIdx(nidx)
    neutral_n.SetFormalCharge(0)
    neutral_n.SetNumExplicitHs(expected_n_h + 1)
    neutral_n.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None

    from .anion_parent_selection import (
        AnionicCenterFact, AnionicParentCandidate, anion_parent_frontier,
    )
    from .constants import ROOTS
    from .parent_selection import graph_parent_facts, principal_context

    context = principal_context(neutral)
    if context is None:
        return None
    principal_class = context[0]
    center = AnionicCenterFact(
        atom_ids=frozenset((nidx,)), charge_atom_ids=frozenset((nidx,)),
        class_symbol="N", operation="ide", evidence_kind="group",
        source_rule="P-72.2.2.2.3",
        proof="one negative N created by hydron loss from a neutral primary or secondary amine",
    )
    candidates = []
    for parent_arm_index, arm in enumerate(arms):
        parent_length = len(arm)
        if parent_length not in ROOTS or parent_length > 60:
            return None
        other_arms = tuple(path for i, path in enumerate(arms) if i != parent_arm_index)
        if len(other_arms) > 1:
            return None
        root = ROOTS[parent_length]
        # P-14.3 permits omission only when attachment is unambiguous. A
        # C1/C2 acyclic parent has one constitution; from C3 onward the amine
        # attachment position must be stated (P-62.2.1.2 examples).
        attachment_locant = "" if parent_length <= 2 else "-1-"
        parent = root + "an" + attachment_locant + "aminide"
        if other_arms:
            other_length = len(other_arms[0])
            if other_length not in ROOTS or other_length > 60:
                return None
            prefix = "N-" + ROOTS[other_length] + "yl"
            name = prefix + parent
        else:
            name = parent
        locants = {idx: str(position) for position, idx in enumerate(arm, 1)}
        facts = graph_parent_facts(
            neutral, arm, locants, principal_class=principal_class,
            suffix_centers=(nidx,), suffix_locants=("1",), prefixes=(),
            kind="chain", name=name,
        )
        if facts is None:
            return None
        candidate = AnionicParentCandidate(
            identity="amine-parent-chain:" + ":".join(map(str, sorted(arm))),
            parent_atom_ids=frozenset(set(arm) | {nidx}),
            considered_charge_atoms=frozenset((nidx,)), parent_centers=(center,),
            suffix_key=(0,), parent_facts=facts,
            proof="amine-derived aminide suffix on a source-selected acyclic amine parent chain",
        )
        candidates.append({"candidate": candidate, "arm": arm,
                           "locants": locants, "name": name})
    if not candidates:
        return None
    frontier = anion_parent_frontier(mol, [row["candidate"] for row in candidates])
    frontier_ids = {id(candidate) for candidate in frontier}
    selected = [row for row in candidates if id(row["candidate"]) in frontier_ids]
    if not selected or len({row["name"] for row in selected}) != 1:
        return None
    mappings = tuple(sorted({tuple(row["arm"]) for row in selected}))
    chosen = min(selected, key=lambda row: tuple(row["arm"]))
    parent_atoms = frozenset(chosen["arm"])
    roles = [
        ("amine-parent-chain", parent_atoms,
         tuple((idx, chosen["locants"][idx]) for idx in chosen["arm"])),
        ("aminide-nitrogen", frozenset((nidx,)), ()),
    ]
    other_atoms = heavy - parent_atoms - {nidx}
    if other_atoms:
        roles.append(("N-alkyl-prefix", frozenset(other_atoms), ()))
    if (len({idx for _, role_atoms, _ in roles for idx in role_atoms}) != len(heavy)
            or set().union(*(set(role_atoms) for _, role_atoms, _ in roles)) != set(heavy)):
        return None
    return NitrogenAnionComponent(
        selected[0]["name"], heavy,
        tuple((idx, chosen["locants"][idx]) for idx in chosen["arm"]), -1, _RULE,
        ("neutral parent reconstructed by adding one H+ to the mapped N charge center",
         "P-72.2.2.2.3 aminide operation is distinct from azanide/amide naming",
         "all source-eligible parent arms compared through P-72.7 then P-44 facts",
         ("equivalent_parent_mappings", mappings),
         ("selected_parent_arm", tuple(chosen["arm"])),
         ("anionic_center", nidx)),
        tuple(roles), parent_atoms, nidx, mappings,
    )


def carbon_aminide_component(mol, Chem):
    """Construct uniform aminide/aminediide or carbon iminide suffixes.

    Each N must independently have the bonding and H population obtained by
    removing exactly -q hydrons from an ordinary amine/imine. In particular,
    R-N(2-) has one single C-N bond and no H; R-NH(-) and R2-N(-) lose one
    hydron; R=N(-) is an iminide. Charge multiplicity at one N and the number
    of distinct suffix sites are independent facts.

    The mapped neutral precursor supplies the ordinary parent, locants and
    N-substituents. Only its terminal, fully accounted-for suffix is changed.
    Isotopes and configuration must survive this operation without changing
    CIP descriptors. Mixed operations and unproved parents decline explicitly
    at the selector; they cannot be represented by a monoanion fallback.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {1, 6, 7} or a.HasQuery()
                   or a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if not nitrogens or len(nitrogens) > 6:
        return None
    if any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetAtomicNum() != 7):
        return None
    operations = []
    for n in nitrogens:
        q = n.GetFormalCharge()
        neighbors = [a for a in n.GetNeighbors() if a.GetAtomicNum() > 1]
        orders = [mol.GetBondBetweenAtoms(n.GetIdx(), a.GetIdx()).GetBondTypeAsDouble()
                  for a in neighbors]
        if (n.IsInRing() or n.GetIsAromatic() or int(n.GetChiralTag())
                or q not in {-1, -2} or not neighbors
                or any(a.GetAtomicNum() != 6 for a in neighbors)
                or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in n.GetBonds())):
            return None
        hydrogen_count = n.GetTotalNumHs(includeNeighbors=True)
        if (orders == [2] and q == -1 and hydrogen_count == 0
                and n.GetTotalValence() == 2):
            operations.append(('imine', 'iminide', 1))
        elif (len(orders) in {1, 2} and all(order == 1 for order in orders)
                and hydrogen_count == 3 + q - len(orders)
                and n.GetTotalValence() == 3 + q):
            operations.append(('amine', 'aminide' if q == -1 else 'aminediide', -q))
        else:
            return None
    if len(set(operations)) != 1:
        return None
    base_suffix, suffix, hydrons = operations[0]
    neutral = Chem.Mol(mol)
    for n in nitrogens:
        atom = neutral.GetAtomWithIdx(n.GetIdx())
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(n.GetNumExplicitHs() + n.GetNumImplicitHs() + hydrons)
        atom.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(neutral)
    except (ValueError, RuntimeError):
        return None
    if not _same_configuration_descriptors(mol, neutral, Chem):
        return None
    from .api import _engine
    from .constants import MULT
    from .formatting import COMPLEX_MULT
    result, mapping = _engine().name_mapped_result(neutral)
    locants = dict(mapping)
    centers = tuple(n.GetIdx() for n in nitrogens)
    # A neutral name with an amino prefix or an embedded amine does not
    # establish a suffix for that N. Every operated N needs its actual N-site
    # locant in the committed parent, not a count inferred from the name.
    if not result.name or any(not locants.get(i, '').startswith('N') for i in centers):
        return None
    count = len(centers)
    name = result.name
    if count == 1 and base_suffix == 'amine' and name.endswith('aniline'):
        name = name[:-len('aniline')] + 'benzen' + suffix
    else:
        old = (MULT[count] if count > 1 else '') + base_suffix
        if not name.endswith(old):
            return None
        ending = COMPLEX_MULT[count] + '(' + suffix + ')' if count > 1 else suffix
        name = name[:-len(old)] + ending
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    carbon = heavy - set(centers)
    roles = [('carbon-framework', carbon, tuple((i, locants[i]) for i in sorted(carbon) if i in locants))]
    roles.extend((suffix + '-nitrogen', frozenset((i,)), ((i, locants[i]),)) for i in centers)
    return NitrogenAnionComponent(
        name=name, atoms=heavy, atom_locants=tuple(mapping),
        charge=sum(n.GetFormalCharge() for n in nitrogens), rule=_RULE + '/P-82/P-9',
        proof=(('hydron_losses', tuple((i, hydrons) for i in centers)),
               ('neutral_parent', result.name),
               'every operated N is a mapped suffix site of the committed neutral parent',
               'unchanged atoms, bonds, isotope populations and configuration descriptors'),
        component_roles=tuple(roles), parent_atoms=carbon, anionic_center=centers[0],
        anionic_centers=centers, suffix=suffix, preference_issues=result.unresolved)


def _same_configuration_descriptors(original, neutral, Chem):
    """A copied neutral descriptor may be used only if original CIP agrees."""
    if (not any(int(a.GetChiralTag()) for a in original.GetAtoms())
            and not any(int(b.GetStereo()) for b in original.GetBonds())):
        return True
    from rdkit.Chem import rdCIPLabeler
    signatures = []
    for graph in (original, neutral):
        graph = Chem.Mol(graph)
        Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(graph)
        signatures.append((
            tuple((a.GetIdx(), int(a.GetChiralTag()), a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else '')
                  for a in graph.GetAtoms() if int(a.GetChiralTag())),
            tuple((b.GetIdx(), int(b.GetStereo()), b.GetProp('_CIPCode') if b.HasProp('_CIPCode') else '')
                  for b in graph.GetBonds() if int(b.GetStereo()))))
    return signatures[0] == signatures[1]


def nitrogen_iminide_component(mol, Chem):
    """P-72.2.2.2.3 imine-derived monoanions on carbon or pnictogens.

    The carbon case reconstitutes the neutral parent imine on a private graph
    and replaces only its terminal suffix. Carbon and pnictogen groups must
    account for the entire original component, with no competing functions.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 64 or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 15, 33, 51, 83} or a.GetIsotope()
                   or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    if len(nitrogens) != 1:
        return None
    nitrogen = nitrogens[0]
    nidx = nitrogen.GetIdx()
    if (nitrogen.GetFormalCharge() != -1 or nitrogen.GetDegree() != 1
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 0
            or nitrogen.GetBonds()[0].GetBondTypeAsDouble() != 2
            or any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != nidx)):
        return None
    host = nitrogen.GetNeighbors()[0]
    hidx = host.GetIdx()
    all_atoms = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if host.GetAtomicNum() == 6 and all(a.GetAtomicNum() in {6, 7} for a in mol.GetAtoms()):
        neutral = Chem.Mol(mol)
        atom = neutral.GetAtomWithIdx(nidx)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(1)
        atom.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(neutral)
        except (RuntimeError, ValueError):
            return None
        from .api import mol_to_iupac
        name = mol_to_iupac(neutral)
        if not name.endswith('imine') or name.count('imine') != 1:
            return None
        name = name[:-5] + 'iminide'
    elif host.GetAtomicNum() in {15, 33, 51, 83}:
        if (host.IsInRing() or host.GetTotalValence() != 5
                or any(a.GetAtomicNum() not in {6, 7, host.GetAtomicNum()} for a in mol.GetAtoms())
                or sum(a.GetAtomicNum() == host.GetAtomicNum() for a in mol.GetAtoms()) != 1):
            return None
        roots = [a.GetIdx() for a in host.GetNeighbors() if a.GetIdx() != nidx]
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
            return None
        from .functional_replacement_acids import _organyl_units
        seen = {hidx, nidx}
        prefix = _organyl_units(mol, Chem, hidx, roots, seen)
        if prefix is None or frozenset(seen) != all_atoms:
            return None
        parent = {15: 'phosphane', 33: 'arsane', 51: 'stibane', 83: 'bismuthane'}[host.GetAtomicNum()]
        name = prefix + ('-' if prefix else '') + 'lambda5-' + parent[:-1] + 'iminide'
    else:
        return None
    return NitrogenAnionComponent(
        name=name, atoms=all_atoms, atom_locants=(), charge=-1,
        rule='P-72.2.2.2.3/P-41',
        proof=('terminal imine-derived N− retains its formal charge and the whole original component',),
        component_roles=(('imine-parent', all_atoms - {nidx}, ()), ('anionic-imine-nitrogen', frozenset((nidx,)), ())),
        parent_atoms=all_atoms - {nidx}, anionic_center=nidx,
    )
