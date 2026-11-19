"""Mapped esters of ring-nitrogen carboxylic acids.

P-65.1.2.2.2 permits an acid suffix at a heterocyclic nitrogen.  An ester
of that acid is expressed as an organyl word and a ring-N-carboxylate, so
the complete substituted ring, not an N-acyl prefix, carries the suffix.
This producer is limited to a saturated monocyclic host or a uniquely numbered
mancude fused host with a complete ring-attachment witness, and proved
component branches. Other hosts, annotations, or unresolved branches stay with
their existing routes.
"""
from types import SimpleNamespace

from .branch_rules import collect_subtree
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .extended_inventory import selection_inventory
from .formatting import enclose_name_unit
from .heterohost_sulfur import _host_is_mapped_ring_nitrogen
from .mapped_ring_parents import parent_facts, records, ring_systems, skeletons
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .neutral_charge_components import (
    neutral_charge_role_proofs, validate_neutral_charge_role_proofs,
)
from .parent_selection import parent_frontier


_RULE = "P-29.6.1/P-41/P-44/P-65.1.2.2.2/P-65.6.3.2.1/P-65.6.3.3.1"
_FUSED_RULE = _RULE + "/P-14.7.1/P-25.3"


def _carbon_tree_branch(mol, Chem, root, blocked):
    """A saturated carbon-only branch with exact atom ownership."""
    scope = frozenset(collect_subtree(mol, root, blocked))
    if (not scope or blocked in scope
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).IsInRing() for i in scope)
            or any(b.GetBondTypeAsDouble() != 1 for b in iter_bonds(mol)
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= scope)):
        return '', frozenset()
    name, owned = mapped_branch(mol, Chem, root, blocked)
    return (name, owned) if name and owned == scope else ('', frozenset())


def _hydroxyalkyl_branch(mol, Chem, root, blocked):
    """A single unbranched hydroxyalkyl arm with exact atom ownership."""
    scope = frozenset(collect_subtree(mol, root, blocked))
    if not scope or blocked in scope or len(scope) > 10:
        return '', frozenset()
    oxygens = [i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
    if (len(oxygens) != 1 or mol.GetAtomWithIdx(root).GetAtomicNum() != 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8}
                   or mol.GetAtomWithIdx(i).IsInRing() for i in scope)):
        return '', frozenset()
    oxygen = mol.GetAtomWithIdx(oxygens[0])
    if (oxygen.GetDegree() != 1 or oxygen.GetFormalCharge()
            or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
        return '', frozenset()
    if (sum(n.GetIdx() in scope for n in mol.GetAtomWithIdx(root).GetNeighbors()) != 1
            or any(sum(n.GetIdx() in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2
                   for i in scope)
            or any(mol.GetBondBetweenAtoms(i, blocked) is not None
                   for i in scope if i != root)
            or any(b.GetBondTypeAsDouble() != 1 for b in iter_bonds(mol)
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= scope)):
        return '', frozenset()
    name, owned = mapped_branch(mol, Chem, root, blocked)
    return (name, owned) if name and owned == scope else ('', frozenset())


def _proved_ring_branch(mol, Chem, root, blocked):
    """Use only a complete mapped cyclic substituent construction."""
    scope = frozenset(collect_subtree(mol, root, blocked))
    if not scope or blocked in scope or not mol.GetAtomWithIdx(root).IsInRing():
        return '', frozenset()
    candidates = records(mol, Chem, scope=scope, attachment=root)
    if len(candidates) != 1:
        return '', frozenset()
    candidate = candidates[0]
    construction = candidate.construction
    if (not candidate.name or not candidate.annotation_coverage
            or not getattr(candidate, 'hydrogen_budget_proved', False)
            or getattr(candidate, 'preference_issues', ())
            or construction is None or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != scope):
        return '', frozenset()
    return candidate.name, scope


def _parent_branch(mol, Chem, root, host):
    """Construct a bounded one- or two-free-valence parent prefix."""
    bond = mol.GetBondBetweenAtoms(root, host)
    if bond.GetBondTypeAsDouble() == 1:
        if mol.GetAtomWithIdx(root).IsInRing():
            return _proved_ring_branch(mol, Chem, root, host)
        scope = frozenset(collect_subtree(mol, root, host))
        atom = mol.GetAtomWithIdx(root)
        if (atom.GetAtomicNum() == 6 and any(
                b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8
                for b in atom.GetBonds())):
            from .branch.subordinate_carbamoyl import (
                mapped_subordinate_carbamoyl, mapped_subordinate_ring_nitrogen_carbonyl,
            )
            carbamoyl = mapped_subordinate_carbamoyl(mol, Chem, root, host)
            if carbamoyl is not None and carbamoyl[1] == scope:
                return carbamoyl
            pseudoketone = mapped_subordinate_ring_nitrogen_carbonyl(mol, Chem, root, host)
            if pseudoketone is not None and pseudoketone[1] == scope:
                return pseudoketone
        if atom.GetAtomicNum() == 7:
            # A separately senior ring-N carbamate can bear a subordinate,
            # completely owned amido branch under P-66.1.1.4.3.
            from .branch.amido_components import mapped_amido_prefix
            amido = mapped_amido_prefix(mol, Chem, root, host, scope)
            if amido is not None and amido[1] == scope:
                return amido
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs(includeNeighbors=True) == 2:
            children = [a for a in atom.GetNeighbors()
                        if a.GetIdx() != host and a.GetAtomicNum() > 1]
            if len(children) == 1 and children[0].GetAtomicNum() == 7:
                from .amine_components import ordinary_amine
                child = children[0].GetIdx()
                if ordinary_amine(mol, child):
                    name, owned = mapped_branch(mol, Chem, child, root)
                    if name and owned | {root} == scope:
                        return enclose_name_unit(name) + 'methyl', scope
            hydroxyalkyl = _hydroxyalkyl_branch(mol, Chem, root, host)
            if hydroxyalkyl[0]:
                return hydroxyalkyl
        return _carbon_tree_branch(mol, Chem, root, host)
    if bond.GetBondTypeAsDouble() != 2:
        return '', frozenset()
    atom = mol.GetAtomWithIdx(root)
    # A single substituted methylidene carbon has one H, one ring-substituent
    # bond and a double bond to the parent. Its ring substituent is rendered
    # independently with its own complete original-atom mapping.
    neighbors = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != host]
    if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 2
            or atom.GetTotalNumHs() != 1 or len(neighbors) != 1
            or mol.GetBondBetweenAtoms(root, neighbors[0]).GetBondTypeAsDouble() != 1):
        return '', frozenset()
    child, owned = _proved_ring_branch(mol, Chem, neighbors[0], root)
    if not child or root in owned:
        return '', frozenset()
    scope = frozenset(collect_subtree(mol, root, host))
    if owned | {root} != scope:
        return '', frozenset()
    # P-29.6.1 retains benzylidene as the preferred prefix only when the
    # phenyl ring is unsubstituted. Prove the ring graph rather than infer
    # that condition from a rendered prefix string.
    if (len(owned) == 6 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            and mol.GetAtomWithIdx(i).GetIsAromatic() for i in owned)
            and all(sum(n.GetIdx() in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()) == 2
                    for i in owned)
            and sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in owned) == 5):
        return 'benzylidene', scope
    return enclose_name_unit(child) + 'methylidene', scope


def _fused_ring_carboxylate(mol, Chem, ring, host, carbonyl, carbonyl_o,
                            ester_o, oxygen_name, oxygen_atoms, heavy):
    """Use a uniquely numbered, fully mapped mancude fused N-host.

    The ring renderer's N-attachment witness supplies the original hydrogen
    state and every substituent. Its free valence is the same ring N bonded
    to the carboxylate carbon; form the acid suffix from the mapped parent
    and locant instead of changing a finished substituent name.
    """
    if (not mol.GetAtomWithIdx(host).GetIsAromatic()
            or mol.GetAtomWithIdx(host).GetDegree() != 3
            or mol.GetAtomWithIdx(host).GetTotalValence() != 3
            or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            or sum(set(cycle) <= ring for cycle in mol.GetRingInfo().AtomRings()) != 2
            or any(a.GetFormalCharge() for a in iter_atoms(mol))):
        return None
    rows = tuple(skeletons(mol, Chem, ring))
    if len(rows) != 1 or rows[0][7] != 'component-fused':
        return None
    suffix_atoms = frozenset({carbonyl, carbonyl_o})
    organyl_atoms = oxygen_atoms | {ester_o}
    scope = heavy - suffix_atoms - organyl_atoms
    if not ring <= scope:
        return None
    candidates = records(mol, Chem, scope=scope, attachment=host, core_filter=ring)
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    construction = candidate.construction
    mapping = dict(candidate.atom_locants)
    if (candidate.method != 'component-fused' or not candidate.annotation_coverage
            or not candidate.hydrogen_budget_proved or candidate.preference_issues
            or candidate.parent != rows[0][0] or frozenset(candidate.parent_atoms) != ring
            or host not in mapping or tuple(candidate.indicated_h_locants) != (mapping[host],)
            or candidate.added_h_locants or candidate.hydro_locants
            or construction is None or construction.atoms != ring
            or construction.atoms | construction.embedded_atoms != scope):
        return None
    branch_atoms = set()
    for root, anchor, atoms, text in construction.prefix_branch_ownership:
        atoms = frozenset(atoms)
        if (anchor not in ring or root not in atoms or atoms & (ring | branch_atoms)
                or not text or mol.GetBondBetweenAtoms(root, anchor) is None
                or frozenset(collect_subtree(mol, root, anchor)) != atoms):
            return None
        branch_atoms.update(atoms)
    if (branch_atoms != set(construction.embedded_atoms)
            or len(construction.prefix_branch_ownership) != len(construction.prefixes)):
        return None
    host_locant = mapping[host]
    hydride = host_locant + 'H-' + candidate.parent
    atom_locants = tuple(candidate.atom_locants) + ((carbonyl, ''), (carbonyl_o, ''))
    component = MappedComponent(
        parent=f'{hydride}-{host_locant}-carboxylate', atoms=ring | suffix_atoms,
        embedded_atoms=construction.embedded_atoms, atom_locants=atom_locants,
        prefixes=construction.prefixes, systematic_parent=True,
        prefix_branch_ownership=construction.prefix_branch_ownership)
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    roles = (('parent-hydride', ring, tuple(candidate.atom_locants)),
             ('carboxylate', suffix_atoms, atom_locants[-2:]),
             ('O-organyl', organyl_atoms, ()),
             ('parent-substituents', frozenset(branch_atoms), ()))
    owned = [i for _role, atoms, _locants in roles for i in atoms]
    if len(owned) != len(set(owned)) or frozenset(owned) != heavy:
        return None
    return MethodDecision(oxygen_name + ' ' + name, 'functional-class',
        frozenset({carbonyl}), heavy, _FUSED_RULE, atom_locants, component_roles=roles)


def annular_carbamate_carboxylate_decision(mol, Chem, core):
    """Return a fully owned, neutral, unannotated ring-N ester or decline."""
    if (core is None or mol.GetNumHeavyAtoms() > 100 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope()
                   or a.GetNumRadicalElectrons() for a in iter_atoms(mol))
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol))):
        return None
    proofs = neutral_charge_role_proofs(mol)
    if not validate_neutral_charge_role_proofs(mol, proofs):
        return None
    carbonyl, carbonyl_o, ester_o, host = core
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'carbamate'
            or inventory.principal[0].center != carbonyl):
        return None
    if (any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in core)
            or mol.GetAtomWithIdx(carbonyl).IsInRing()
            or mol.GetAtomWithIdx(carbonyl_o).GetDegree() != 1
            or mol.GetAtomWithIdx(ester_o).GetDegree() != 2
            or mol.GetAtomWithIdx(ester_o).IsInRing()):
        return None
    rings = [ring for ring in ring_systems(mol) if host in ring]
    if len(rings) != 1:
        return None
    ring = rings[0]
    # The mapped ring writer can carry tetrahedral descriptors on the chosen
    # host. Unspecified potential configurations remain unspecified and are
    # recorded as an unresolved preferred-name obligation, never guessed.
    potential_stereo = tuple(Chem.FindPotentialStereo(mol))
    if (any(item.type != Chem.StereoType.Atom_Tetrahedral or item.centeredOn not in ring
            for item in potential_stereo)
            or any(int(a.GetChiralTag()) and (a.GetIdx() not in ring or a.GetChiralTag() not in
                   (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
                   for a in iter_atoms(mol))):
        return None
    from .naming_result import SelectionLimitation
    stereo_issues = ((SelectionLimitation(
        'stereochemistry', 'P-91.2.1.2.1/P-91.2.2',
        'One or more potential stereogenic units have no source configuration.'),)
        if any(item.specified != Chem.StereoSpecified.Specified for item in potential_stereo)
        else ())
    oxygen_roots = [a.GetIdx() for a in mol.GetAtomWithIdx(ester_o).GetNeighbors()
                    if a.GetIdx() != carbonyl]
    if len(oxygen_roots) != 1:
        return None
    oxygen_name, oxygen_atoms = _carbon_tree_branch(mol, Chem, oxygen_roots[0], ester_o)
    if not oxygen_name:
        return None
    suffix_atoms = frozenset({carbonyl, carbonyl_o})
    organyl_atoms = oxygen_atoms | {ester_o}
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if mol.GetAtomWithIdx(host).GetIsAromatic():
        # The fused writer's existing hydrogen/annotation proof deliberately
        # excludes stereogenic systems until it gains its own stereo witness.
        if potential_stereo:
            return None
        return _fused_ring_carboxylate(mol, Chem, ring, host, carbonyl, carbonyl_o,
                                       ester_o, oxygen_name, oxygen_atoms, heavy)
    if (not 3 <= len(ring) <= 10
            or sum(set(cycle) <= ring for cycle in mol.GetRingInfo().AtomRings()) != 1
            or any(b.GetBondTypeAsDouble() != 1 for b in iter_bonds(mol)
                   if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= ring)
            or not _host_is_mapped_ring_nitrogen(mol, host, ring, carbonyl)):
        return None
    candidates = []
    skeleton_rows = tuple(skeletons(mol, Chem, ring))
    for skeleton in skeleton_rows:
        parent, sequence, locant_values = skeleton[:3]
        if frozenset(sequence) != ring or not skeleton[5]:
            return None
        mapping = dict(zip(sequence, locant_values))
        prefixes, branches, roles, ownership = [], set(), [], []
        valid = True
        for idx in sequence:
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = neighbor.GetIdx()
                if root in ring or root == carbonyl:
                    continue
                text, atoms = _parent_branch(mol, Chem, root, idx)
                if (not text or root not in atoms or branches & atoms
                        or atoms & (ring | suffix_atoms | organyl_atoms)):
                    valid = False
                    break
                prefixes.append((mapping[idx], text))
                branches.update(atoms)
                roles.append(('parent-substituent', atoms, ()))
                ownership.append((root, idx, atoms, text))
            if not valid:
                break
        if (not valid or ring | suffix_atoms | organyl_atoms | branches != heavy
                or any(not (set(proof.atoms) | {proof.attachment_atom}) <= branches
                       for proof in proofs)):
            return None
        host_locant = mapping[host]
        atom_locants = tuple(mapping.items()) + ((carbonyl, ''), (carbonyl_o, ''))
        component = MappedComponent(
            parent=f'{parent}-{host_locant}-carboxylate',
            atoms=ring | suffix_atoms, embedded_atoms=frozenset(branches),
            atom_locants=atom_locants, prefixes=tuple(prefixes),
            systematic_parent=True, prefix_branch_ownership=tuple(ownership))
        component_name = decorate_component(mol, Chem, component)
        if not component_name:
            return None
        name = oxygen_name + ' ' + component_name
        record = SimpleNamespace(parent=parent, parent_atoms=tuple(sequence),
            atom_locants=tuple(mapping.items()), principal_centers=frozenset({carbonyl}),
            suffix_locants=(host_locant,), prefixes=component.prefixes, name=name,
            indicated_h_locants=(), method='annular-carboxylate', construction=component)
        facts = parent_facts(mol, record, (9, 1))
        all_roles = (('parent-hydride', ring, tuple(mapping.items())),
                     ('carboxylate', suffix_atoms, atom_locants[-2:]),
                     ('O-organyl', organyl_atoms, ()), *roles)
        owned = [i for _role, atoms, _locants in all_roles for i in atoms]
        if len(owned) != len(set(owned)) or frozenset(owned) != heavy:
            return None
        signature = (parent, host_locant, tuple(sorted(prefixes)))
        candidates.append((record, facts, all_roles, signature))
    if not candidates:
        return None
    if any(row[1] is None for row in candidates):
        # The shared graph comparator currently validates nested charge units
        # only across single-bond cuts. The independently proved methylidene
        # branch above can also enclose one, but supplies no new comparator.
        # It is usable only if every original ring-numbering witness has the
        # same parent, suffix site and substituted-site expressions, leaving
        # no ranking decision to infer from the names.
        if (not proofs or len({row[3] for row in candidates}) != 1
                or any(not any(
                    (set(proof.atoms) | {proof.attachment_atom}) <= set(atoms)
                    and mol.GetBondBetweenAtoms(root, anchor).GetBondTypeAsDouble() == 2
                    for root, anchor, atoms, _text
                    in row[0].construction.prefix_branch_ownership)
                    for row in candidates for proof in proofs)):
            return None
        frontier = candidates
    else:
        frontier = parent_frontier(candidates, lambda row: row[1])
    if not frontier or len({row[0].name for row in frontier}) != 1:
        return None
    record, _facts, roles, _signature = frontier[0]
    return MethodDecision(record.name, 'functional-class', record.principal_centers,
                          heavy, _RULE, record.construction.atom_locants,
                          component_roles=roles, preference_issues=stereo_issues)
