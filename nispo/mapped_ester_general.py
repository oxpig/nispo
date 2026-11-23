"""P-65.6 constructions from mapped ester arms and acid-side parents."""

from collections import defaultdict
from dataclasses import dataclass, replace

from .branch_rules import collect_subtree
from .component_mapping import fully_covered, mapped_general_branch
from .constants import ROOTS
from .extended_inventory import selection_inventory
from .formatting import enclose_name_unit, format_prefixes, prefix_alphabetization_key
from .functional_inventory import CharacteristicGroup
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .parent_selection import locant_key, parent_frontier, replace_parent_facts


_RULE = 'P-41/P-65.2.1.1/P-65.6.3.3'


@dataclass(frozen=True, slots=True)
class EsterArm:
    group: CharacteristicGroup
    root: int
    name: str
    atoms: frozenset[int]


def mapped_ester_arm(mol, Chem, group):
    """Bind the carbonyl, bridge O and one complete alcohol-derived branch."""
    if group.kind not in {'ester', 'carbamate'}:
        return None
    center = mol.GetAtomWithIdx(group.center)
    oxygen = mol.GetAtomWithIdx(group.ligand)
    bridge = mol.GetBondBetweenAtoms(group.center, group.ligand)
    if (center.GetAtomicNum() != 6 or center.IsInRing() or oxygen.GetAtomicNum() != 8
            or oxygen.GetDegree() != 2 or oxygen.GetFormalCharge()
            or bridge is None or bridge.GetBondTypeAsDouble() != 1):
        return None
    roots = [atom for atom in oxygen.GetNeighbors() if atom.GetIdx() != group.center]
    if len(roots) != 1:
        return None
    root = roots[0]
    # P-65.6.3.4.1 includes pseudoesters attached to cyclic nitrogen.
    if (root.GetAtomicNum() != 6 and not (root.GetAtomicNum() == 7 and root.IsInRing())
            or mol.GetBondBetweenAtoms(root.GetIdx(), group.ligand).GetBondTypeAsDouble() != 1):
        return None
    root = root.GetIdx()
    name, atoms = mapped_general_branch(mol, Chem, root, group.ligand)
    if not name and group.kind == 'ester' and mol.GetAtomWithIdx(root).IsInRing():
        from .component_mapping import _complete_rooted_ring

        # The strict branch route can decline a polyoxygenated ring when one
        # of its side chains requires an independently mapped general prefix.
        scope = frozenset(collect_subtree(mol, root, group.ligand))
        if root in scope and group.ligand not in scope:
            name = _complete_rooted_ring(mol, Chem, root, scope)
            if name:
                atoms = scope
    atoms = frozenset(atoms)
    if (not name or any(char.isspace() for char in name) or root not in atoms
            or atoms & set(group.atoms)
            or {(idx, bond.GetOtherAtomIdx(idx)) for idx in atoms
                for bond in mol.GetAtomWithIdx(idx).GetBonds()
                if bond.GetOtherAtomIdx(idx) not in atoms} != {(root, group.ligand)}):
        return None
    return EsterArm(group, root, name, atoms)


def ester_organyl_words(mol, Chem, arms, locants):
    """Cite each mapped alcohol component before the acid-derived anion."""
    from .organyl_prefix_components import multiplied_organyl_prefix
    from .polyacid_ester_methods import _organyl_identity

    if len(arms) == 1:
        return arms[0].name
    species = defaultdict(list)
    for arm, locant in zip(arms, locants, strict=True):
        identity = _organyl_identity(mol, Chem, arm.atoms, arm.group.ligand, arm.root)
        if not identity:
            return ''
        species[arm.name].append((arm, locant, identity))
    if len(species) > 1 and len(set(locants)) != len(locants):
        # Unlike groups at indistinguishable geminal sites require a
        # separate assignment proof.
        return ''
    words = []
    for name in sorted(species, key=prefix_alphabetization_key):
        members = species[name]
        if len({identity for _arm, _locant, identity in members}) != 1:
            return ''
        if len(members) > 1:
            arm = members[0][0]
            name = (multiplied_organyl_prefix(mol, arm.root, arm.atoms, name, len(members))
                    or format_prefixes([('', name)] * len(members), omit_locants=True))
        if len(species) > 1:
            locant_text = ','.join(sorted((locant for _arm, locant, _identity in members),
                                          key=locant_key))
            name = locant_text + '-' + (enclose_name_unit(name) if name[:1].isdigit() else name)
        words.append(name)
    return ' '.join(words)


def _carbamate_nitrogen(mol, group):
    return next(i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)


def carboxylate_for_formate(mol, Chem):
    """Use the mapped acid-side carbon framework of a substituted formate."""
    inventory = selection_inventory(mol)
    groups = inventory.principal
    if (not inventory.supported or len(groups) != 1 or groups[0].kind != 'ester'
            or groups[0].owner < 0):
        return None
    decision = mapped_ester_decision(mol, Chem)
    if decision is not None and decision.principal_centers == {groups[0].center}:
        return decision
    return None


def _ring_constructions(mol, Chem, arms, scope):
    from .mapped_ring_parents import parent_facts, records, ring_systems
    from .stereochemistry import assigned_atom_stereochemistry

    for core in ring_systems(mol):
        candidates = []
        for arm in arms:
            group = arm.group
            if group.kind == 'carbamate':
                # An exocyclic N-carbamate uses an external carboxylate
                # suffix on the ring nitrogen, which stays a skeletal atom.
                nitrogen = _carbamate_nitrogen(mol, group)
                group = replace(group, kind='ester', owner=nitrogen,
                                atoms=tuple(i for i in group.atoms if i != nitrogen))
            if group.owner in core:
                candidates.append((group, arm))
        if not candidates:
            continue
        # Match the bounded monoester route's ordinary atom-stereo scope.
        # Leave a pseudoasymmetric acid-side ring to its existing stereo
        # writer until this general route can compare those constructions.
        if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core) and any(
                i in core and descriptor in {"r", "s"}
                for i, descriptor in assigned_atom_stereochemistry(mol, Chem, include_pseudo=True)):
            continue
        # All independent arms on this frame form a polybasic acid
        # component. Retain monoesters as general continuations if that
        # construction is unavailable.
        batches = [tuple(candidates)] if len(candidates) > 1 else []
        batches.extend((candidate,) for candidate in candidates)
        for batch in batches:
            groups, selected_arms = zip(*batch)
            rows = records(mol, Chem, scope=scope, core_filter=core,
                           principal=groups, ester_arms=selected_arms, general_branches=True)
            if rows:
                for row in rows:
                    yield row, parent_facts(mol, row, (9,)), selected_arms
                if len(batch) == len(candidates):
                    break


def _chain_constructions(mol, Chem, arms, inventory):
    from .complete_functional_parent import _render_functional_candidate
    from .mapped_principal_general import _carbon_parent_plans
    from .parent_rules import _ester_acyl_parent_paths

    ordinary = {arm.group: arm for arm in arms if arm.group.kind == 'ester'}
    plans = []
    if len(ordinary) > 1:
        plans.extend((path, covered, external) for path, covered, external in
                     _carbon_parent_plans(mol, tuple(ordinary), inventory) if len(covered) > 1)
    for arm in ordinary.values():
        group = arm.group
        if group.owner < 0:
            paths = ((group.center,),) if mol.GetAtomWithIdx(group.center).GetTotalNumHs() else ()
        elif mol.GetAtomWithIdx(group.owner).IsInRing():
            continue
        else:
            paths = tuple((group.center, *path) for path in
                          _ester_acyl_parent_paths(mol, Chem, group.center, group.owner))
        plans.extend((path, (group,), False) for path in paths)
    for path, groups, external in plans:
        selected = tuple(ordinary[group] for group in groups)
        organyl_atoms = frozenset().union(*(arm.atoms for arm in selected))
        group_atoms = {i for group in groups for i in group.atoms}
        if (len(organyl_atoms) != sum(len(arm.atoms) for arm in selected)
                or organyl_atoms.intersection(group_atoms | set(path))):
            continue
        skipped = (group_atoms - {group.center for group in groups}) | organyl_atoms
        row = _render_functional_candidate(
            mol, Chem, groups, 'ester', path, external, False, skipped,
            tuple((arm.name, arm.atoms) for arm in selected), mapped_ring_prefixes=True,
            charge_role_proofs=inventory.charge_role_proofs, general_branches=True,
            locanted_acid_fallback=True)
        if row is not None:
            facts = row.parent_selection
            yield row, (replace_parent_facts(facts, principal_class=(9,))
                        if facts is not None else None), selected


def _acyclic_carbamate_constructions(mol, Chem, arms):
    from .carbamate_methods import render_acyclic_carbamate

    for arm in arms:
        group = arm.group
        if group.kind != 'carbamate':
            continue
        nitrogen = _carbamate_nitrogen(mol, group)
        if mol.GetAtomWithIdx(nitrogen).IsInRing():
            continue
        parent_atoms = frozenset(group.atoms) - {group.ligand}
        branches, owned = [], set()
        for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
            root = atom.GetIdx()
            if root == group.center:
                continue
            text, atoms = mapped_general_branch(mol, Chem, root, nitrogen)
            if not text:
                text, atoms = _mapped_imidoyl_carbamate_branch(mol, Chem, root, nitrogen)
            if (not text or any(c.isspace() for c in text)
                    or set(atoms) & (owned | parent_atoms | arm.atoms | {group.ligand})
                    or frozenset(collect_subtree(mol, root, nitrogen)) != atoms):
                break
            branches.append(text)
            owned.update(atoms)
        else:
            if not fully_covered(mol, parent_atoms | {group.ligand}, owned | arm.atoms):
                continue
            body = render_acyclic_carbamate(mol, Chem, nitrogen, parent_atoms, branches)
            if body:
                yield arm, arm.name + ' ' + body


def _mapped_imidoyl_carbamate_branch(mol, Chem, root, nitrogen):
    """Map a carbon-substituted C(=NH) arm on the carbamate nitrogen."""
    from .component_mapping import _complete_rooted_ring, _general_branch_stereo_is_cited

    carbon = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, nitrogen)
    if (cut is None or carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3
            or carbon.GetFormalCharge() or carbon.GetTotalNumHs(includeNeighbors=True)
            or cut.GetBondTypeAsDouble() != 1 or cut.GetStereo()):
        return '', frozenset()
    bonds = {(b.GetOtherAtom(carbon).GetAtomicNum(), b.GetBondTypeAsDouble()): b
             for b in carbon.GetBonds() if b.GetOtherAtomIdx(root) != nitrogen}
    if bonds.keys() != {(6, 1), (7, 2)}:
        return '', frozenset()
    imino_bond = bonds[7, 2]
    imino = imino_bond.GetOtherAtom(carbon)
    if (imino.GetDegree() != 1 or imino.GetFormalCharge()
            or imino.GetTotalNumHs(includeNeighbors=True) != 1 or imino_bond.GetStereo()):
        return '', frozenset()
    ligand = bonds[6, 1].GetOtherAtom(carbon)
    idx = ligand.GetIdx()
    ligand_scope = frozenset(collect_subtree(mol, idx, root))
    if {root, nitrogen, imino.GetIdx()} & ligand_scope:
        return '', frozenset()
    name, owned = mapped_general_branch(mol, Chem, idx, root)
    if not name and ligand.IsInRing():
        name = _complete_rooted_ring(mol, Chem, idx, ligand_scope)
        owned = ligand_scope
    if (not name or owned != ligand_scope
            or not _general_branch_stereo_is_cited(mol, Chem, ligand_scope, name)):
        return '', frozenset()
    return enclose_name_unit(name) + 'methanimidoyl', ligand_scope | {root, imino.GetIdx()}


def _linear_ester_path(mol, group, organyl_atoms):
    """Find the complete unbranched carbon parent and its connector bond."""
    path = [group.center]
    blocked = frozenset(group.atoms) | organyl_atoms
    while len(path) <= 64:
        current = mol.GetAtomWithIdx(path[-1])
        outside = [a for a in current.GetNeighbors()
                   if a.GetIdx() not in blocked and a.GetIdx() not in path and a.GetAtomicNum() > 1]
        if len(outside) != 1:
            return (), None
        following = outside[0]
        bond = mol.GetBondBetweenAtoms(path[-1], following.GetIdx())
        if bond.GetBondTypeAsDouble() != 1:
            return (), None
        if following.IsInRing() or following.GetAtomicNum() != 6:
            return tuple(path), following.GetIdx()
        path.append(following.GetIdx())
    return (), None


def _two_ester_connector(mol, Chem, scope, ports, paths):
    """Render an oxygen bridge or a symmetric aromatic connector with two ports."""
    from .mapped_ring_parents import ring_systems, skeletons

    if not set(ports) <= scope:
        return ''
    if len(scope) == 1 and ports[0] == ports[1]:
        atom = mol.GetAtomWithIdx(ports[0])
        return 'oxy' if atom.GetAtomicNum() == 8 and atom.GetDegree() == 2 else ''
    if len(set(ports)) != 2:
        return ''
    cores = ring_systems(mol)
    if len(cores) != 1 or not cores[0] <= scope:
        return ''
    core = cores[0]
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
        return ''
    outer = scope - core
    if not outer and set(ports) <= core:
        hosts, arms = ports, ''
    elif outer == set(ports):
        hosts = []
        for port, path in zip(ports, paths, strict=True):
            oxygen = mol.GetAtomWithIdx(port)
            if oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2:
                return ''
            host = next(a.GetIdx() for a in oxygen.GetNeighbors() if a.GetIdx() != path[-1])
            if host not in core or mol.GetBondBetweenAtoms(port, host).GetBondTypeAsDouble() != 1:
                return ''
            hosts.append(host)
        arms = 'bis(oxy)'
    else:
        return ''
    if len(set(hosts)) != 2:
        return ''
    orientations = set()
    for parent, seq, locants, *_ in skeletons(mol, Chem, core):
        if parent not in {'benzene', 'naphthalene', 'anthracene', 'phenanthrene'}:
            continue
        mapping = dict(zip(seq, locants, strict=True))
        left, right = (mapping[idx] for idx in hosts)
        if left.isdigit() and right.isdigit():
            orientations.add((parent, left, right))
    options = []
    for parent, left, right in orientations:
        if (parent, right, left) not in orientations:
            continue
        positions = tuple(sorted((left, right), key=locant_key))
        text = ','.join(positions)
        body = f'{text}-phenylene' if parent == 'benzene' else f'{parent}-{text}-diyl'
        options.append((tuple(map(locant_key, positions)), body + arms))
    if not options:
        return ''
    best = min(key for key, _ in options)
    names = {text for key, text in options if key == best}
    return names.pop() if len(names) == 1 else ''


def _multiplicative_diester(mol, Chem, arms, scope):
    """Compare the two complete linear acid components before monoester rendering."""
    if (len(arms) != 2 or any(arm.group.kind != 'ester' for arm in arms)
            or arms[0].name != arms[1].name
            or any(a.GetFormalCharge() or a.GetChiralTag() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.GetBondDir() for b in mol.GetBonds())):
        return None
    paths, ports = zip(*(_linear_ester_path(mol, arm.group, arm.atoms) for arm in arms), strict=True)
    length = len(paths[0])
    if length < 2 or length not in ROOTS or len(paths[1]) != length:
        return None
    units = tuple(frozenset(path) | frozenset(arm.group.atoms) | arm.atoms
                  for path, arm in zip(paths, arms, strict=True))
    if units[0] & units[1]:
        return None
    connector = scope - units[0] - units[1]
    for unit, port, path in zip(units, ports, paths, strict=True):
        if {(i, n.GetIdx()) for i in unit for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in unit} != {(path[-1], port)}:
            return None
    bridge = _two_ester_connector(mol, Chem, connector, ports, paths)
    if not bridge:
        return None
    organyls = ester_organyl_words(mol, Chem, arms, ('1', "1'"))
    if not organyls:
        return None
    parent = 'acetate' if length == 2 else ROOTS[length] + 'anoate'
    link = bridge if bridge == 'oxy' else enclose_name_unit(bridge)
    name = f"{organyls} {length},{length}'-{link}di{parent}"
    locants = tuple((idx, str(i) + prime)
                    for path, prime in zip(paths, ('', "'"), strict=True)
                    for i, idx in enumerate(path, 1))
    return MethodDecision(
        name, 'multiplicative', frozenset(arm.group.center for arm in arms),
        scope, 'P-51.3.1/P-65.6.3.3.2.2.1/P-65.6.3.3.4.1', locants,
        component_roles=(('connector', connector, ()), *(
            ('ester', unit, tuple(row for row in locants if row[0] in unit)) for unit in units)),
        preference_issues=(SelectionLimitation(
            'ester-parent-and-method', _RULE,
            'The equivalent ester arms and their connector are mapped. '
            'Complete preferred-name comparison remains open.'),))


def mapped_multiplicative_diester(mol, Chem, groups, scope):
    """Construct two mapped equivalent ester units for the P-51 comparison."""
    if (len(groups) != 2 or any(group.kind != 'ester' for group in groups)
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms())):
        return None
    arms = tuple(arm for group in groups if (arm := mapped_ester_arm(mol, Chem, group)) is not None)
    return _multiplicative_diester(mol, Chem, arms, scope)


def mapped_ester_decision(mol, Chem, *, require_parent_proof=False):
    """Deliver a class-9 construction with exact ownership and open method scope."""
    if mol.GetStereoGroups() or any(
            atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.HasQuery()
            for atom in mol.GetAtoms()):
        return None
    inventory = selection_inventory(mol)
    if require_parent_proof and any(
            atom.GetAtomicNum() == 16 and atom.GetDegree() == 3
            and sum(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8
                    for b in atom.GetBonds()) == 1 for atom in mol.GetAtoms()):
        # The strict direct consumer has no source-qualified proof choosing
        # the preferred sulfinyl branch vocabulary. Keep its established
        # full delivery until that component evidence is supplied.
        return None
    facts = inventory.source_group_facts
    if (not inventory.supported or not facts
            or any(fact is None or fact.class_rank is None for fact in facts)
            or min(fact.class_rank for fact in facts) != 9):
        return None
    groups = tuple(group for group, fact in zip(inventory.groups, facts, strict=True)
                   if fact.class_rank == 9)
    # The ordinary carboxylic ester remains senior to carbonate arms.
    # Carbonates can be owned by a mapped prefix on either ester component.
    if any(group.kind == 'ester' for group in groups):
        groups = tuple(group for group in groups if group.kind != 'carbonate')
    if any(group.kind not in {'ester', 'carbamate'} for group in groups):
        return None
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    if not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs):
        return None
    arms = tuple(arm for group in groups if (arm := mapped_ester_arm(mol, Chem, group)) is not None)
    scope = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    candidates = [row for row in (*_ring_constructions(mol, Chem, arms, scope),
                                 *_chain_constructions(mol, Chem, arms, inventory))
                  if row[0].construction is not None
                  and fully_covered(mol, row[0].construction.atoms,
                                    row[0].construction.embedded_atoms)]
    if candidates:
        maximum = max(len(arms) for _row, _facts, arms in candidates)
        candidates = [row for row in candidates if len(row[2]) == maximum]
        # Compare acid-side parents only after site coverage. A carboxylic
        # acid component is senior to a carbonic-acid derivative (P-42).
        acid_maximum = max(sum(a.group.kind == 'ester' for a in row[2]) for row in candidates)
        candidates = [row for row in candidates
                      if sum(a.group.kind == 'ester' for a in row[2]) == acid_maximum]
        if require_parent_proof:
            from .mapped_ring_parents import ring_systems

            owners = {(_carbamate_nitrogen(mol, group) if group.kind == 'carbamate' else group.owner)
                      for group in groups}
            frames = [core for core in ring_systems(mol) if owners <= core]
            # All source sites on one complete ring frame give the maximum
            # possible site count. P-65's corresponding acid is that ring
            # carboxylate, rather than a substituted formate. Other rings own
            # no competing principal sites. Numbering is compared below.
            if (len(frames) != 1 or len(arms) != len(groups)
                    or any(facts is None for _row, facts, _arms in candidates)):
                return None
            centers = {group.center for group in groups}
            candidates = [option for option in candidates
                          if frozenset(option[0].parent_atoms) == frames[0]
                          and {arm.group.center for arm in option[2]} == centers]
            if not candidates:
                return None
        if all(facts is not None for _row, facts, _arms in candidates):
            candidates = parent_frontier(candidates, lambda row: row[1])
        if require_parent_proof and len({row.name for row, _facts, _arms in candidates}) != 1:
            return None
        row, _facts, selected = min(candidates, key=lambda option: option[0].name)
        name, locants = row.name, row.atom_locants
        acid_atoms = scope - frozenset().union(*(a.atoms | {a.group.ligand} for a in selected))
        locants = tuple((i, locant) for i, locant in locants if i in acid_atoms)
    else:
        if any(group.kind == 'ester' for group in groups):
            if len(groups) == len(arms) == 1 and groups[0].owner >= 0 and mol.GetAtomWithIdx(groups[0].owner).IsInRing():
                from .spiro_component_rules import mapped_spiro_ester

                return mapped_spiro_ester(mol, Chem, arms[0], scope)
            return None
        carbamates = list(_acyclic_carbamate_constructions(mol, Chem, arms))
        if not carbamates:
            return None
        if require_parent_proof and (len(groups) != 1 or len(arms) != 1
                                    or len({name for _arm, name in carbamates}) != 1):
            return None
        if require_parent_proof:
            from .mapped_ring_parents import ring_systems

            if any(sum(set(ring) <= core for ring in mol.GetRingInfo().AtomRings()) > 1
                   for core in ring_systems(mol)):
                # The fixed acyclic carbamate parent does not itself select
                # a preferred fused parent in an N/O ligand. That branch
                # comparison remains with the established full route.
                return None
        arm, name = min(carbamates, key=lambda option: option[1])
        selected, locants = (arm,), ()
        acid_atoms = scope - arm.atoms - {arm.group.ligand}
    rule = _RULE
    if len(selected) == 1 and not require_parent_proof:
        from .amino_acid_esters import retained_amino_acid_ester

        arm = selected[0]
        retained = retained_amino_acid_ester(mol, Chem, arm.group, arm.name, arm.atoms)
        if retained:
            name = retained
            rule += '/P-103.2.6'
    roles = (('acid', acid_atoms, locants), *(
        ('O-organyl', arm.atoms | {arm.group.ligand}, ()) for arm in selected))
    centers = frozenset(arm.group.center for arm in selected)
    issues = (SelectionLimitation(
        'ester-parent-and-method', _RULE,
        'The ester components are mapped. Complete parent and method comparison remains open.'),)
    if centers != {group.center for group in groups}:
        issues += (SelectionLimitation(
            'principal-group', _RULE,
            'The name cites a class-9 function. Citation of the other class-9 sites remains open.'),)
    return MethodDecision(
        name, 'functional-class', centers,
        scope, rule, locants, component_roles=roles,
        preference_issues=issues)
