"""Mapped P-15.4/P-51.4 construction across standard skeletal elements.

The admitted universe is a neutral acyclic tree with a chain containing every
skeletal heteroatom and every principal site. Remaining branches are saturated
carbon trees or ordinary halogens directly attached to a parent carbon. Named
multiatom heterounits are counted by the shared eligibility
proof, not by the number of heteroatoms. Fixed heterochain numbering precedes
suffix and multiple-bond locants; detachable prefixes are compared afterwards.

Homogeneous ordinary suffixes are admitted, as is a bounded monoacid with
subordinate hydroxy groups on a C/O chain with at most methyl carbon branches.
Functional-class sites, other mixed classes, nonstandard valence and unresolved
annotation comparisons remain with their existing producers. No parser
participates in construction.
"""

from dataclasses import dataclass
from itertools import combinations

from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_branch
from .constants import MULT, ROOTS
from .formatting import (attach_suffix, complete_parent_name, multiplied_suffix,
                         prefix_alphabetization_key, unsaturated_parent)
from .replacement_eligibility import acyclic_replacement_eligibility, _TERMINAL_ELEMENTS


RULE = "P-51.4.1.1-P-51.4.1.4/P-15.4.1.1-P-15.4.1.2/P-15.4.3/P-44.3/P-44.4/P-45.2"
# P-15.4.1.2 element order; these are standard bonding numbers, not guessed
# oxidation states. Halogens remain outside this acyclic skeletal vocabulary.
_ELEMENTS = (
    (8, "oxa", 2), (16, "thia", 2), (34, "selena", 2), (52, "tellura", 2),
    (7, "aza", 3), (15, "phospha", 3), (33, "arsa", 3),
    (51, "stiba", 3), (83, "bisma", 3),
    (14, "sila", 4), (32, "germa", 4), (50, "stanna", 4), (82, "plumba", 4),
    (5, "bora", 3), (13, "alumina", 3), (31, "galla", 3),
    (49, "inda", 3), (81, "thalla", 3),
)
_VALENCES = {number: valence for number, _term, valence in _ELEMENTS} | {6: 4}
_HALOGENS = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}
_NON_SKELETAL_HETEROUNITS = frozenset((6, *_HALOGENS))


@dataclass(frozen=True, slots=True)
class ChainSuffix:
    kind: str
    center: int
    atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class StandardHeterochainCandidate:
    decision: object
    path: tuple[int, ...]
    suffixes: tuple[ChainSuffix, ...]
    prefixes: tuple
    parent_identity: tuple
    eligibility: object


def _neighbors(mol, idx):
    return tuple(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                 if n.GetAtomicNum() > 1)


def _plain_carbon_halogen(mol, idx):
    atom = mol.GetAtomWithIdx(idx)
    neighbors = _neighbors(mol, idx)
    return (not atom.GetIsotope() and len(neighbors) == 1
            and mol.GetAtomWithIdx(neighbors[0]).GetAtomicNum() == 6
            and mol.GetBondBetweenAtoms(idx, neighbors[0]).GetBondTypeAsDouble() == 1)


def _suffixes(mol):
    """Recognize complete ordinary local sites without an element whitelist.

    Global ordinary inventories intentionally exclude Si/Ge/etc. These exact
    graph roles let the replacement producer recognize a suffix without
    replacing those skeletal atoms or changing their chemical interpretation.
    """
    result, occupied = [], set()
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        idx = carbon.GetIdx()
        neighbors = _neighbors(mol, idx)
        double_o = tuple(i for i in neighbors if mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                         and mol.GetBondBetweenAtoms(idx, i).GetBondTypeAsDouble() == 2)
        triple_n = tuple(i for i in neighbors if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                         and mol.GetBondBetweenAtoms(idx, i).GetBondTypeAsDouble() == 3)
        if double_o:
            if len(double_o) != 1:
                return None
            single_o = tuple(i for i in neighbors if mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                             and mol.GetBondBetweenAtoms(idx, i).GetBondTypeAsDouble() == 1)
            single_n = tuple(i for i in neighbors if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                             and mol.GetBondBetweenAtoms(idx, i).GetBondTypeAsDouble() == 1)
            if len(single_o) == 1 and not single_n:
                kind, atoms = "acid", frozenset((idx, *double_o, *single_o))
            elif len(single_n) == 1 and not single_o:
                kind, atoms = "amide", frozenset((idx, *double_o, *single_n))
            elif (not single_o and not single_n and len(neighbors) == 3
                  and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                          and mol.GetBondBetweenAtoms(idx, i).GetBondTypeAsDouble() == 1
                          for i in neighbors if i not in double_o)):
                kind, atoms = "ketone", frozenset((idx, *double_o))
            else:
                return None
            if kind in {"acid", "amide"}:
                from types import SimpleNamespace
                from .acyl_replacement_methods import _typed_group
                if _typed_group(mol, SimpleNamespace(kind=kind, center=idx, atoms=atoms)) is None:
                    return None
                if kind == "acid" and any(mol.GetAtomWithIdx(i).GetAtomicNum()
                                          not in {6, 14, 32, 50, 82}
                                          for i in neighbors if i not in atoms):
                    return None
                if kind == "amide" and any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                                            for i in neighbors if i not in atoms):
                    return None
            result.append(ChainSuffix(kind, idx, atoms))
            occupied.update(atoms - {idx})
        elif triple_n:
            if len(triple_n) != 1:
                return None
            from types import SimpleNamespace
            from .acyl_replacement_methods import _typed_group
            atoms = frozenset((idx, *triple_n))
            if _typed_group(mol, SimpleNamespace(kind="nitrile", center=idx, atoms=atoms)) is None:
                return None
            # P-66.5.1.1.3 requires the external carbonitrile operation for
            # CN directly attached to a heteroatom; this producer spells the
            # embedded-chain nitrile suffix only.
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   for i in neighbors if i not in atoms):
                return None
            result.append(ChainSuffix("nitrile", idx, atoms))
            occupied.update(triple_n)
    for atom in mol.GetAtoms():
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        neighbors = _neighbors(mol, idx)
        if (idx in occupied or number not in {8, 16} or len(neighbors) != 1
                or atom.GetTotalNumHs(includeNeighbors=True) != 1
                or atom.GetTotalValence() != 2
                or mol.GetAtomWithIdx(neighbors[0]).GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(idx, neighbors[0]).GetBondTypeAsDouble() != 1):
            continue
        result.append(ChainSuffix("alcohol" if number == 8 else "thiol",
                                  neighbors[0], frozenset((idx, neighbors[0]))))
    kinds = {group.kind for group in result}
    if len(kinds) > 1:
        # P-41 ranks an ordinary carboxylic acid above alcohol. The mixed
        # construction below has its own closed graph and ownership scope;
        # this local site census alone does not authorize other mixed classes.
        if kinds != {"acid", "alcohol"} or sum(g.kind == "acid" for g in result) != 1:
            return None
        return tuple(result)
    # The interaction of fixed replacement numbering and omission for two
    # terminal acyl suffixes has its own existing proof/limitations. Do not
    # silently generalize that spelling from the single-site source examples.
    if len(result) > 1 and result[0].kind in {"acid", "amide", "nitrile"}:
        return None
    return tuple(result)


def _paths(adjacency):
    leaves = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) == 1)
    for start, finish in combinations(leaves, 2):
        pending = [(start, (start,))]
        while pending:
            idx, path = pending.pop()
            if idx == finish:
                yield path
                break
            pending.extend((other, path + (other,)) for other in adjacency[idx]
                           if len(path) == 1 or other != path[-2])


def _orientation_facts(mol, path, groups):
    elements = tuple(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path)
    hetero = tuple(i for i, number in enumerate(elements, 1) if number != 6)
    by_element = tuple(tuple(i for i, number in enumerate(elements, 1) if number == element)
                       for element, _term, _valence in _ELEMENTS)
    suffix = tuple(sorted(path.index(group.center) + 1 for group in groups))
    ene, yne = [], []
    for loc, (a, b) in enumerate(zip(path, path[1:]), 1):
        order = mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
        if order == 2:
            ene.append(loc)
        elif order == 3:
            yne.append(loc)
    return (hetero, by_element, suffix, tuple(sorted(ene + yne)), tuple(ene)), tuple(ene), tuple(yne)


def _parent_text(mol, path, groups, ene, yne):
    parts = []
    for number, term, _valence in _ELEMENTS:
        locants = [str(i) for i, idx in enumerate(path, 1)
                   if mol.GetAtomWithIdx(idx).GetAtomicNum() == number]
        if locants:
            multiplier = MULT.get(len(locants)) if len(locants) > 1 else ""
            if multiplier is None:
                return None
            parts.append(",".join(locants) + "-" + multiplier + term)
    root = ROOTS.get(len(path))
    if root is None or any(len(locs) > 1 and len(locs) not in MULT for locs in (ene, yne)):
        return None
    base = complete_parent_name(unsaturated_parent(root, list(ene), list(yne)))
    parent = "-".join(parts) + base if base else None
    if parent and groups:
        suffix = {"acid": "oic acid", "amide": "amide", "nitrile": "nitrile",
                  "alcohol": "ol", "ketone": "one", "thiol": "thiol"}[groups[0].kind]
        # P-65/P-66 retain explicit locants on these fixed heterochains.
        suffix = multiplied_suffix(len(groups), suffix)
        parent = attach_suffix(parent, suffix,
                               sorted(path.index(group.center) + 1 for group in groups))
    return parent


def _candidate(mol, Chem, path, groups, proof, heavy, external, halogens,
               subordinate_hydroxy=()):
    locants = tuple((idx, str(i)) for i, idx in enumerate(path, 1))
    mapping, owned = dict(locants), set(path) | set(external)
    prefixes, roles = [], [("replacement_parent", frozenset(path), locants)]
    hydroxy_hosts = {next(iter(group.atoms - {group.center})): group.center
                     for group in subordinate_hydroxy}
    if external:
        roles.append(("principal_suffix", external, ()))
    for idx in path:
        for other in _neighbors(mol, idx):
            if other in owned:
                continue
            if mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1:
                return None
            if other in hydroxy_hosts:
                if hydroxy_hosts[other] != idx:
                    return None
                prefixes.append((mapping[idx], "hydroxy"))
                roles.append(("hydroxy_prefix", frozenset((other,)), ()))
                owned.add(other)
                continue
            if other in halogens:
                # The halogen is a substituent, not a replacement heterounit.
                # Keep its original atom separate from the numbered parent;
                # labelled atoms and non-carbon hosts are declined upstream.
                if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
                    return None
                prefixes.append((mapping[idx], _HALOGENS[mol.GetAtomWithIdx(other).GetAtomicNum()]))
                roles.append(("halogen_prefix", frozenset((other,)), ()))
                owned.add(other)
                continue
            if subordinate_hydroxy and (
                    mol.GetAtomWithIdx(other).GetAtomicNum() != 6
                    or _neighbors(mol, other) != (idx,)
                    or mol.GetAtomWithIdx(other).GetTotalNumHs(includeNeighbors=True) != 3):
                # In the mixed-function scope, atom coverage alone cannot
                # prove an arbitrary carbon branch's preferred spelling.
                # Direct methyl has a source-fixed simple prefix.
                return None
            text, atoms = mapped_branch(mol, Chem, other, idx)
            if (not text or not atoms or atoms & owned
                    or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms)
                    or any(b.GetBondTypeAsDouble() != 1 for i in atoms
                           for b in mol.GetAtomWithIdx(i).GetBonds()
                           if b.GetOtherAtomIdx(i) in atoms)):
                return None
            prefixes.append((mapping[idx], text))
            roles.append(("carbon_branch", frozenset(atoms), ()))
            owned.update(atoms)
    if not fully_covered(mol, owned, ()):
        return None
    _rank, ene, yne = _orientation_facts(mol, path, groups)
    parent = _parent_text(mol, path, groups, ene, yne)
    if not parent:
        return None
    name = decorate_component(mol, Chem, MappedComponent(
        parent, locants, frozenset(path), tuple(prefixes), heavy - frozenset(path)))
    if not name:
        return None
    from .naming_methods import MethodDecision
    rule = RULE + "/P-41/P-35.2.1" if subordinate_hydroxy else RULE
    decision = MethodDecision(name, "skeletal-replacement",
        frozenset(group.center for group in groups), heavy, rule, locants,
        tuple(unit.atoms for unit in proof.heterounits), tuple(roles))
    identity = (tuple(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in path),
                tuple(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                      for a, b in zip(path, path[1:])),
                tuple(sorted((group.kind, mapping[group.center]) for group in groups)))
    return StandardHeterochainCandidate(decision, path, groups, tuple(prefixes), identity, proof)


def _mixed_acid_hydroxy_scope(mol, Chem, groups, heavy):
    """Prove local C/O site roles; the caller still enumerates parent paths.

    This is deliberately narrower than general mixed-function nomenclature:
    one ordinary acid, carbon-bound OH, isolated ether O atoms, and no
    isotope or potential stereochemistry. No preference proof is attached.
    """
    acids = tuple(group for group in groups if group.kind == "acid")
    hydroxy = tuple(group for group in groups if group.kind == "alcohol")
    if len(acids) != 1 or not hydroxy or len(acids) + len(hydroxy) != len(groups):
        return None
    if any(atom.GetAtomicNum() not in {6, 8} or atom.HasQuery() or atom.GetIsotope()
           or int(atom.GetChiralTag()) for atom in mol.GetAtoms()):
        return None
    if any(bond.HasQuery() or int(bond.GetStereo()) or int(bond.GetBondDir())
           for bond in mol.GetBonds()):
        return None
    acid = acids[0]
    hydroxys = frozenset(i for group in hydroxy for i in group.atoms if i != group.center)
    acid_oxygens = acid.atoms - {acid.center}
    if (len(acid_oxygens) != 2 or len(hydroxys) != len(hydroxy)
            or acid_oxygens & hydroxys):
        return None
    for bond in mol.GetBonds():
        atoms = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if bond.GetBondTypeAsDouble() != 1 and not (
                bond.GetBondTypeAsDouble() == 2 and acid.center in atoms
                and len(atoms & acid_oxygens) == 1):
            return None
    ether_oxygens = {i for i in heavy - acid_oxygens - hydroxys
                     if mol.GetAtomWithIdx(i).GetAtomicNum() == 8}
    if len(ether_oxygens) < 4 or any(
            len(_neighbors(mol, idx)) != 2
            or any(mol.GetAtomWithIdx(neighbor).GetAtomicNum() != 6
                   for neighbor in _neighbors(mol, idx))
            or mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True)
            for idx in ether_oxygens):
        return None
    try:
        if Chem.FindPotentialStereo(mol):
            return None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    frontier = inventory.source_frontier
    recognized = {(group.kind, group.center, frozenset(group.atoms)) for group in groups}
    inventoried = {(group.kind, group.center, frozenset(group.atoms)) for group in inventory.groups}
    if (not inventory.supported or len(groups) != len(inventory.groups)
            or recognized != inventoried
            or frontier.status != "selected" or frontier.unknown_pairs or frontier.cycle
            or len(frontier.groups) != 1 or frontier.groups[0].kind != "acid"
            or frontier.groups[0].center != acid.center
            or frozenset(frontier.groups[0].atoms) != acid.atoms):
        return None
    return acids, hydroxy, hydroxys


def standard_heterochain_frontier(mol, Chem):
    """Return all source-tied complete candidates, or decline this producer.

    Leaf extension cannot reduce covered sites or skeletal heteroatoms in this
    tree universe. Requiring all of both closes those two upper bounds. Thus
    the leaf-pair frontier contains every source-maximum parent in this scope.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetRingInfo().NumRings()
            or mol.GetStereoGroups()):
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (sum(mol.GetAtomWithIdx(i).GetAtomicNum() not in _NON_SKELETAL_HETEROUNITS for i in heavy) < 4
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetAtomicNum() == 1 and a.GetIsotope()
                   or a.GetAtomicNum() > 1 and a.GetAtomicNum() not in _VALENCES
                   and a.GetAtomicNum() not in _HALOGENS
                   for a in mol.GetAtoms())):
        return None
    if any(a.GetTotalValence() != (_VALENCES.get(a.GetAtomicNum(), 1))
           for a in mol.GetAtoms() if a.GetAtomicNum() > 1):
        return None
    halogens = frozenset(i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in _HALOGENS)
    if any(not _plain_carbon_halogen(mol, i) for i in halogens):
        return None
    # P-51.4 excludes atoms belonging to an acid/ester/amide principal group.
    # Ordinary bonding number alone does not distinguish e.g. phosphonites
    # or boronic esters from skeletal P/B. Their element-specific producers
    # own that classification, including cases without a terminal OH group.
    if any(a.GetAtomicNum() in {5, 13, 31, 49, 81, 15, 33, 51, 83}
           and any(n.GetAtomicNum() in {7, 8, 16, 34, 52} for n in a.GetNeighbors())
           for a in mol.GetAtoms()):
        return None
    groups = _suffixes(mol)
    if groups is None:
        return None
    subordinate_hydroxy, hydroxys = (), frozenset()
    if len({group.kind for group in groups}) > 1:
        mixed = _mixed_acid_hydroxy_scope(mol, Chem, groups, heavy)
        if mixed is None:
            return None
        groups, subordinate_hydroxy, hydroxys = mixed
    external = frozenset(i for group in groups for i in group.atoms if i != group.center)
    if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           for i in external):
        return None
    skeleton = heavy - external - halogens - hydroxys
    # Ordinary amines have their own principal-group and multiplicative rules.
    # Nitrogen is skeletal here only under a proved senior suffix class.
    if not groups and any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in skeleton):
        return None
    adjacency = {i: frozenset(j for j in _neighbors(mol, i) if j in skeleton) for i in skeleton}
    if (not skeleton or sum(map(len, adjacency.values())) != 2 * (len(skeleton) - 1)
            or len(Chem.GetMolFrags(mol)) != 1):
        return None
    # Non-suffix multiple bonds are admitted only between ordinary carbons.
    if any(b.GetBondTypeAsDouble() not in {1, 2, 3}
           or b.GetBondTypeAsDouble() != 1
           and not (b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 6)
           for b in mol.GetBonds()
           if b.GetBeginAtomIdx() in skeleton and b.GetEndAtomIdx() in skeleton):
        return None
    all_hetero = {i for i in skeleton if mol.GetAtomWithIdx(i).GetAtomicNum() != 6}
    hosts = {group.center for group in groups}
    paths = []
    for path in _paths(adjacency):
        if (not (all_hetero | hosts) <= set(path)
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in _TERMINAL_ELEMENTS
                       for i in (path[0], path[-1]))):
            continue
        proof = acyclic_replacement_eligibility(mol, path,
            principal_group_atoms=external, has_principal_suffix=bool(groups))
        if proof.eligible:
            orders = [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                      for a, b in zip(path, path[1:])]
            rank = (-len(path), -sum(o > 1 for o in orders), -orders.count(2))
            paths.append((rank, path, proof))
    if not paths:
        return None
    best = min(rank for rank, _path, _proof in paths)
    orientations = [(path, proof) for rank, p, proof in paths if rank == best
                    for path in (p, tuple(reversed(p)))]
    fixed = min(_orientation_facts(mol, path, groups)[0] for path, _proof in orientations)
    candidates = []
    for path, proof in orientations:
        if _orientation_facts(mol, path, groups)[0] != fixed:
            continue
        candidate = _candidate(mol, Chem, path, groups, proof, heavy, external, halogens,
                               subordinate_hydroxy)
        if candidate is None:
            return None  # An unrendered maximum is never discarded for a lower one.
        candidates.append(candidate)
    if len({item.parent_identity for item in candidates}) != 1:
        return None
    if len(candidates) > 1:
        # An exact automorphism of the fully annotated input proves all
        # original-graph comparison facts tie, including unspecified units.
        # The locant labels bind the parent map; user atom-map numbers do not
        # participate. This is graph evidence, not equality of output names.
        numbered_graphs = []
        for candidate in candidates:
            copy = Chem.Mol(mol)
            for atom in copy.GetAtoms():
                atom.SetAtomMapNum(0)
            for idx, locant in candidate.decision.atom_locants:
                copy.GetAtomWithIdx(idx).SetAtomMapNum(int(locant))
            numbered_graphs.append(Chem.MolToSmiles(copy, isomericSmiles=True))
        if len(set(numbered_graphs)) == 1:
            return tuple(candidates) if len({c.decision.name for c in candidates}) == 1 else None
    from .parent_annotations import isotope_seniority, stereo_seniority, stereo_identity
    def retain(candidates, key):
        if len(candidates) <= 1:
            return candidates
        ranked = [(key(candidate), candidate) for candidate in candidates]
        if any(value is None for value, _candidate in ranked):
            return None
        best = min(value for value, _candidate in ranked)
        return [candidate for value, candidate in ranked if value == best]
    # Each source criterion is applied only to the surviving frontier. A
    # later unknown fact cannot invalidate an already decisive earlier one.
    candidates = retain(candidates, lambda c: isotope_seniority(mol, dict(c.decision.atom_locants)))
    if candidates is None:
        return None
    if len(candidates) > 1:
        identities = [stereo_identity(mol, dict(c.decision.atom_locants)) for c in candidates]
        same_stereo = identities[0] is not None and all(x == identities[0] for x in identities)
        if not same_stereo:
            stereos = [(stereo_seniority(mol, dict(c.decision.atom_locants)), c) for c in candidates]
            if any(value is None for value, _candidate in stereos):
                return None
            if len({value[0] if value else () for value, _candidate in stereos}) > 1:
                return None
            best_stereo = min(value for value, _candidate in stereos)
            candidates = [c for value, c in stereos if value == best_stereo]
    def prefix_rank(candidate):
        ordered = sorted(candidate.prefixes,
                         key=lambda row: (prefix_alphabetization_key(row[1]), int(row[0])))
        return (-len(ordered), tuple(sorted(int(loc) for loc, _ in ordered)),
                tuple(int(loc) for loc, _ in ordered))
    candidates = retain(candidates, prefix_rank)
    from .parent_annotations import substituent_isotopes, isotope_numbering_key, stereo_numbering
    for operation in (substituent_isotopes, isotope_numbering_key):
        candidates = retain(candidates, lambda c: operation(mol, dict(c.decision.atom_locants)))
        if candidates is None:
            return None
    candidates = retain(candidates, lambda c: prefix_alphabetization_key(c.decision.name)[:3])
    # P-45.6 configuration resolves a choice only after isotopic and
    # alphanumerical criteria have tied.
    candidates = retain(candidates, lambda c: stereo_numbering(mol, dict(c.decision.atom_locants)))
    if candidates is None:
        return None
    if len({c.decision.name for c in candidates}) != 1:
        return None
    return tuple(candidates)
