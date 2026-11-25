"""P-66.3.3.3. Choose the acid parent before assigning N and N′."""

from dataclasses import dataclass, replace

from .acid_derivative_methods import _acid_component, _acyl
from .branch_rules import collect_subtree
from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_general_branch, nitrogen_locant
from .extended_inventory import selection_inventory
from .functional_class_queries import acylhydrazide_query, amidine_query, only_lower_substituents
from .hydrazide_components import _replace_acid_ending
from .naming_methods import MethodDecision
from .parent_selection import compare_parents, graph_parent_facts, locant_key, locant_set
from .rooted_carbon_parents import grouped_site_orders, hydride_numberings, skeleton_prefixes, substituent_numbering_key


@dataclass(frozen=True)
class AcylArm:
    center: int
    nitrogen: int
    acid_class: tuple
    atoms: frozenset
    parent: MappedComponent
    prefix: str
    identity: str
    facts: object


def _chalcogen_arm(mol, Chem, center, nitrogen, owner, oxos):
    atom = mol.GetAtomWithIdx(center)
    if atom.GetIsotope() or atom.GetChiralTag() or any(mol.GetAtomWithIdx(i).GetIsotope() for i in oxos):
        return None
    atoms = frozenset(collect_subtree(mol, center, nitrogen))
    skeleton_scope = atoms - {center, *oxos}
    stem = {16: 'sulf', 34: 'selen', 52: 'tellur'}[atom.GetAtomicNum()]
    stem += 'on' if len(oxos) == 2 else 'in'
    options = []
    for hydride, skeleton, frame, pi_key in hydride_numberings(mol, Chem, owner, skeleton_scope):
        mapping = hydrogen_bindings(mol, skeleton)
        prefixes = skeleton_prefixes(mol, Chem, mapping, skeleton_scope)
        if prefixes is None:
            continue
        pos = skeleton[owner]
        omit = (len(skeleton) == 1 or pos == '1' and not prefixes
                and (hydride in {'benzene', 'ethane'} or hydride.startswith('cyclo') and hydride.endswith('ane')))
        base = hydride + ('' if omit else '-' + pos + '-') + stem
        parent = MappedComponent(base + 'ohydrazide', tuple(mapping.items()), frozenset(mapping),
                                 tuple(prefixes), atoms - mapping.keys(), systematic_parent=True)
        prefix = decorate_component(mol, Chem, replace(parent, parent=base + 'yl'))
        key = substituent_numbering_key(mol, mapping, prefixes)
        if prefix and key is not None:
            options.append(((frame, locant_key(pos), pi_key, key), parent, prefix))
    if not options:
        return None
    best = min(k for k, _, _ in options)
    winners = {prefix: parent for key, parent, prefix in options if key == best}
    if len(winners) != 1:
        return None
    prefix, parent = winners.popitem()
    # Compare only the parent skeleton of this acid residue. Exclude both
    # hydrazine nitrogens and every other residue from its acid-parent facts.
    skeletal = {i: loc for i, loc in parent.atom_locants if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
    ordered = sorted(atoms)
    inverse = {a: i for i, a in enumerate(ordered)}
    acid = Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in atoms:
            acid.RemoveAtom(i)
    cap = acid.AddAtom(Chem.Atom(8))
    acid.AddBond(inverse[center], cap, Chem.BondType.SINGLE)
    acid = acid.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None
    local = {inverse[i]: loc for i, loc in skeletal.items()}
    facts = graph_parent_facts(acid, tuple(local), local, principal_class=(7, 0),
        suffix_centers=(inverse[center],), suffix_locants=(skeletal[owner],),
        prefixes=parent.prefixes, kind='ring' if mol.GetAtomWithIdx(owner).IsInRing() else 'chain',
        name=prefix, allow_annotations=True)
    identity = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), isomericSmiles=True)
    return AcylArm(center, nitrogen, (atom.GetAtomicNum(), -len(oxos)), atoms, parent, prefix, identity, facts)


def _arm(mol, Chem, center, nitrogen):
    atom = mol.GetAtomWithIdx(center)
    oxos, owners = [], []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetIdx() == nitrogen:
            continue
        if other.GetAtomicNum() == 8 and other.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 2:
            oxos.append(other.GetIdx())
        elif other.GetAtomicNum() == 6 and bond.GetBondTypeAsDouble() == 1:
            owners.append(other.GetIdx())
        elif other.GetAtomicNum() != 1:
            return None
    if atom.GetAtomicNum() != 6:
        if len(oxos) in {1, 2} and len(owners) == 1:
            return _chalcogen_arm(mol, Chem, center, nitrogen, owners[0], oxos)
        return None
    if len(oxos) != 1 or len(owners) > 1:
        return None
    acid = _acid_component(mol, Chem, center, nitrogen, allow_mapped_ring=True,
                           allow_mapped_chain=True, hydrazide=True)
    if acid is None or acid.construction is None:
        return None
    name = _replace_acid_ending(acid.construction.parent)
    prefix = _acyl(acid.name)
    if not name or not prefix:
        return None
    parent = replace(acid.construction, parent=name)
    if mol.GetAtomWithIdx(oxos[0]).GetIsotope():
        # The selected hydrazide owns its carbonyl oxygen. The temporary acid
        # writer had localized the mass at its acid ending.
        mapping = dict(parent.atom_locants)
        mapping[oxos[0]] = 'O'
        parent = replace(parent, atom_locants=tuple(mapping.items()),
                         local_isotope_atoms=parent.local_isotope_atoms - {oxos[0]})
    return AcylArm(center, nitrogen, (6,), acid.atoms, parent, prefix, acid.identity, acid.parent_facts)


def _senior(arms):
    acid_class = min(arm.acid_class for arm in arms)
    candidates = [arm for arm in arms if arm.acid_class == acid_class]
    for arm in candidates:
        for other in candidates:
            if other is arm or arm.identity == other.identity:
                continue
            if compare_parents(arm.facts, other.facts).preference != -1:
                break
        else:
            yield arm


def _imidoyl_prefix(mol, Chem, center, nitrogen, imine):
    """R–C(=NH)– as the acid-derived substituent of an ordinary hydrazide."""
    end = mol.GetAtomWithIdx(imine)
    if end.GetDegree() != 1 or end.GetIsotope() or end.GetTotalNumHs() != 1:
        return mapped_general_branch(mol, Chem, center, nitrogen)
    proxy = Chem.RWMol(mol)
    proxy.ReplaceAtom(imine, Chem.Atom(8))
    proxy = proxy.GetMol()
    try:
        Chem.SanitizeMol(proxy)
    except (ValueError, RuntimeError):
        return '', frozenset()
    acid = _acid_component(proxy, Chem, center, nitrogen, allow_mapped_ring=True,
                           allow_mapped_chain=True, hydrazide=True)
    if acid is None:
        return '', frozenset()
    component = acid.construction
    retained = {'formic acid': 'methanimidoyl', 'acetic acid': 'ethanimidoyl',
                'benzoic acid': 'benzenecarboximidoyl'}
    word = retained.get(component.parent)
    if word is None:
        for suffix, ending in [('carboxylic acid', 'carboximidoyl'), ('oic acid', 'imidoyl')]:
            if component.parent.endswith(suffix):
                word = component.parent.removesuffix(suffix) + ending
                break
    if not word:
        return '', frozenset()
    return decorate_component(mol, Chem, replace(component, parent=word)), acid.atoms


def _polyacylhydrazide(mol, Chem, pairs, matches, attachments):
    """Locate one polyacid residue before interpreting either end of each N–N bond."""
    nitrogens = frozenset().union(*pairs)
    if (len(nitrogens) != 2 * len(pairs) or len(attachments) <= len(pairs)
            or mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                            for a in mol.GetAtoms())):
        return None
    # Cutting all N–residue bonds once exposes the polyacid and every
    # terminal residue. A component touching multiple N atoms is never
    # mistaken for a terminal substituent.
    cut = Chem.RWMol(mol)
    ports = []
    for pair in pairs:
        for n in pair:
            atom = mol.GetAtomWithIdx(n)
            if atom.GetTotalValence() != 3 or atom.GetChiralTag():
                return None
            for bond in atom.GetBonds():
                root = bond.GetOtherAtomIdx(n)
                if root in pair or mol.GetAtomWithIdx(root).GetAtomicNum() == 1:
                    continue
                if root in nitrogens or bond.GetBondTypeAsDouble() != 1:
                    return None
                cut.RemoveBond(root, n)
                ports.append((root, n))
    components = [frozenset(f) for f in Chem.GetMolFrags(cut) if not nitrogens.intersection(f)]
    residues = [(comp, [(r, n) for r, n in ports if r in comp]) for comp in components]
    shared = [(comp, at) for comp, at in residues if len(at) > 1]
    if len(shared) != 1:
        return None
    backbone, central = shared[0]
    if (not set(central) <= attachments or len({c for c, _ in central}) != len(pairs)
            or len(central) != len(pairs) or any(sum(n in p for _, n in central) != 1 for p in pairs)):
        return None
    centers = {c for c, _ in attachments}
    oxygen = {c: o for o, c, _, _ in matches}
    if any(mol.GetAtomWithIdx(c).GetAtomicNum() != 6 for c, _ in central):
        return None
    inventory = selection_inventory(mol)
    if (any(g.center not in centers and g.seniority <= (12, 0) for g in inventory.groups) or
            not inventory.supported and not only_lower_substituents(mol, nitrogens | centers |
                                                                    {o for o, _, _, _ in matches})):
        return None
    branches = []
    for comp, at in residues:
        if comp == backbone:
            continue
        if len(at) != 1:
            return None
        root, n = at[0]
        if (root, n) in attachments:
            arm = _arm(mol, Chem, root, n)
            if arm is None:
                return None
            word, owned = arm.prefix, arm.atoms
        elif mol.GetAtomWithIdx(root).GetAtomicNum() == 6:
            word, owned = mapped_general_branch(mol, Chem, root, n)
        else:
            return None
        if not word or owned != comp:
            return None
        branches.append((n, word))
    sites = []
    partners = {n: other for pair in pairs for n in pair for other in pair if other != n}
    for c, n in central:
        owners = [a.GetIdx() for a in mol.GetAtomWithIdx(c).GetNeighbors() if a.GetIdx() not in {oxygen[c], n}]
        if len(owners) != 1 or mol.GetAtomWithIdx(owners[0]).GetAtomicNum() != 6:
            return None
        sites.append((c, owners[0], n, partners[n]))
    return _polyhydrazide_name(mol, Chem, sites, oxygen, backbone, branches)


def _polyhydrazide_name(mol, Chem, sites, oxygen, backbone, branches):
    from .formatting import attach_suffix, multiplied_suffix

    external = len(sites) > 2 or any(mol.GetAtomWithIdx(owner).IsInRing() for _, owner, _, _ in sites)
    hosts = [owner if external else c for c, owner, _, _ in sites]
    required = set(hosts)
    centers = frozenset(c for c, _, _, _ in sites)
    framework = backbone - {oxygen[c] for c in centers}
    if external:
        framework -= centers
    suffix = multiplied_suffix(len(sites), 'carbohydrazide' if external else 'hydrazide')
    if not suffix:
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    frames = []
    for hydride, skeleton, frame, pi in hydride_numberings(mol, Chem, next(iter(required)), framework, extended=True):
        if not required <= skeleton.keys():
            continue
        positions = [skeleton[i] for i in hosts]
        if not all(p.isdecimal() for p in positions):
            continue
        frames.append(((frame, locant_set(positions), pi), hydride, skeleton, positions))
    options = []
    for rank, hydride, skeleton, positions in sorted(frames, key=lambda row: row[0]):
        if options and rank != options[0][0][0]:
            break
        base_mapping = hydrogen_bindings(mol, skeleton)
        prefixes = skeleton_prefixes(mol, Chem, base_mapping, framework)
        if prefixes is None:
            continue
        parent = attach_suffix(hydride, suffix, sorted(positions, key=locant_key) if external else ())
        if not external and hydride == 'ethane':
            parent = 'oxalohydrazide'
        for order in grouped_site_orders(sites, positions):
            mapping = dict(base_mapping)
            for loc, group in order:
                for j, (c, _, n, other) in enumerate(group):
                    mapping[n] = 'N' + "'" * (2*j) + loc
                    mapping[other] = 'N' + "'" * (2*j + 1) + loc
                    mapping[oxygen[c]] = 'O' + "'" * j + loc
                    if external:
                        mapping[c] = 'C' + "'" * j + loc
            mapping = hydrogen_bindings(mol, mapping)
            items = prefixes + [(mapping[n], word) for n, word in branches]
            key = substituent_numbering_key(mol, mapping, items)
            if key is None:
                continue
            component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping),
                tuple((nitrogen_locant(loc), word) for loc, word in items), scope - mapping.keys(),
                systematic_parent=True)
            name = decorate_component(mol, Chem, component)
            if name:
                options.append(((rank, key), name, mapping))
    if not options:
        return None
    best = min(k for k, _, _ in options)
    winners = {name: m for k, name, m in options if k == best}
    if len(winners) == 1:
        name, mapping = winners.popitem()
        return MethodDecision(name, 'substitutive', centers, scope,
                              'P-66.3.3.3/P-66.3.3.2/P-44', tuple(mapping.items()))
    return None


def acylhydrazide_decision(mol, Chem):
    matches = mol.GetSubstructMatches(acylhydrazide_query(Chem))
    attachments = {(c, n) for _, c, n, _ in matches}
    pairs = {frozenset((n, other)) for _, _, n, other in matches}
    if len(pairs) > 1:
        return _polyacylhydrazide(mol, Chem, pairs, matches, attachments)
    if not pairs:
        return None
    pair = pairs.pop()
    imidoyls = {(c, n): im for im, c, n in mol.GetSubstructMatches(amidine_query(Chem))
                if n in pair and im not in pair and mol.GetAtomWithIdx(c).GetAtomicNum() == 6}
    if len(attachments) + len(imidoyls) < 2:
        return None
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    centers = {c for c, _ in attachments}
    inventory = selection_inventory(mol)
    if any(g.center not in centers and g.seniority <= (12, 0) for g in inventory.groups):
        return None
    known = pair | centers | {o for o, _, _, _ in matches} | set(imidoyls.values())
    if not inventory.supported and not only_lower_substituents(mol, known):
        return None
    arms, branches = [], []
    occupied = set(hydrogen_bindings(mol, dict.fromkeys(pair, '')))
    for n in pair:
        atom = mol.GetAtomWithIdx(n)
        if atom.GetTotalValence() != 3 or atom.GetChiralTag():
            return None
        for bond in atom.GetBonds():
            root = bond.GetOtherAtomIdx(n)
            other = mol.GetAtomWithIdx(root)
            if root in pair or other.GetAtomicNum() == 1:
                continue
            if bond.GetBondTypeAsDouble() != 1:
                return None
            if (root, n) in attachments:
                arm = _arm(mol, Chem, root, n)
                if arm is None:
                    return None
                arms.append(arm)
                atoms = arm.atoms
            elif other.GetAtomicNum() == 6:
                if (root, n) in imidoyls:
                    word, atoms = _imidoyl_prefix(mol, Chem, root, n, imidoyls[root, n])
                else:
                    word, atoms = mapped_general_branch(mol, Chem, root, n)
                if not word:
                    return None
                branches.append((n, word))
            else:
                return None
            if occupied & atoms:
                return None
            occupied.update(atoms)
    scope = frozenset(range(mol.GetNumAtoms()))
    if occupied != scope:
        return None
    options = []
    for arm in _senior(arms):
        nmap = {n: ('N' if n == arm.nitrogen else "N'") for n in pair}
        candidate = arm.parent
        mapping = hydrogen_bindings(mol, dict(candidate.atom_locants) | nmap)
        substitutions = [(other.nitrogen, other.prefix) for other in arms if other is not arm] + branches
        prefixes = candidate.prefixes + tuple((nmap[n], prefix) for n, prefix in substitutions)
        component = replace(candidate, atom_locants=tuple(mapping.items()), atoms=frozenset(mapping),
                            embedded_atoms=scope - mapping.keys(), prefixes=prefixes, nitrogen_site_locants=True)
        key = substituent_numbering_key(mol, mapping, prefixes)
        name = decorate_component(mol, Chem, component)
        if key is not None and name:
            options.append((key, name, mapping, arm.center))
    if not options:
        return None
    best = min(key for key, *_ in options)
    winners = {name: (mapping, center) for key, name, mapping, center in options if key == best}
    if len(winners) != 1:
        return None
    name, (mapping, center) = winners.popitem()
    return MethodDecision(name, 'substitutive', frozenset({center}), scope,
                          'P-66.3.3.3/P-44', tuple(mapping.items()))
