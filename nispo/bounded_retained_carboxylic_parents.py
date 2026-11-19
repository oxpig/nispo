"""Graph-closed retained carboxylic acid parents for derivative methods.

These fallbacks cover a deliberately small parent universe when the general
cross-route frontier does not yet decide an isolated capped acid fragment.
They return the ordinary mapped construction, not a rewritten whole name.
"""

from collections import defaultdict
from dataclasses import dataclass

from .extended_inventory import selection_inventory
from .molecule_iteration import iter_atoms, iter_bonds


@dataclass(frozen=True, slots=True)
class BoundedAcidParent:
    name: str
    atom_locants: tuple
    construction: object


def substituted_benzene_paths(mol, ring, owner, substituent_labels):
    """Rank both directions from locant 1 using original substituent roles."""
    adjacent = [n.GetIdx() for n in mol.GetAtomWithIdx(owner).GetNeighbors()
                if n.GetIdx() in ring]
    if len(adjacent) != 2:
        return ()
    paths = []
    for second in adjacent:
        path = [owner, second]
        while len(path) < 6:
            remaining = [n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                         if n.GetIdx() in ring and n.GetIdx() != path[-2]]
            if len(remaining) != 1 or remaining[0] in path:
                return ()
            path.append(remaining[0])
        if mol.GetBondBetweenAtoms(path[-1], path[0]) is None:
            return ()
        by_kind = defaultdict(list)
        for locant, atom_id in enumerate(path, 1):
            for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors():
                if neighbor.GetIdx() in substituent_labels:
                    by_kind[substituent_labels[neighbor.GetIdx()]].append(locant)
        key = (tuple(sorted(loc for positions in by_kind.values() for loc in positions)),
               tuple((kind, tuple(by_kind[kind])) for kind in sorted(by_kind)))
        paths.append((key, tuple(path)))
    best = min(key for key, _ in paths)
    return tuple(path for key, path in paths if key == best)


def amino_acetic_acid_parent(mol, Chem):
    """Prove an acetic parent and a mapped methyl(cycloalkyl)amino branch."""
    if mol.GetStereoGroups() or any(
        a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
        or int(a.GetChiralTag()) or a.GetAtomicNum() == 1
        for a in iter_atoms(mol)
    ) or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 2
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'acid'
            or sum(g.kind == 'amine' for g in inventory.groups) != 1):
        return None
    group = inventory.principal[0]
    alpha = mol.GetAtomWithIdx(group.owner)
    if (alpha.GetAtomicNum() != 6 or alpha.IsInRing() or alpha.GetIsAromatic()
            or alpha.GetTotalNumHs() != 2 or alpha.GetDegree() != 2
            or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1):
        return None
    roots = [n.GetIdx() for n in alpha.GetNeighbors() if n.GetIdx() != group.center]
    if len(roots) != 1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum() != 7:
        return None
    from .component_mapping import mapped_branch
    from .retained_acid_rules import _methyl_cycloamino_branch

    root = roots[0]
    text, owned = mapped_branch(mol, Chem, root, group.owner)
    if (not text or set(owned) | set(group.atoms) | {group.owner} !=
            {a.GetIdx() for a in iter_atoms(mol)}
            or not _methyl_cycloamino_branch(
                mol, Chem, root, group.owner, owned, text, inventory)):
        return None
    return _mapped_acetic_acid_parent(mol, Chem, group, root, text, owned)


def alkoxycarbonylamino_acetic_acid_parent(mol, Chem):
    """Use the senior carboxylic-acid suffix and a proved carbamate prefix."""
    if mol.GetStereoGroups() or any(
        a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
        or int(a.GetChiralTag()) or a.GetAtomicNum() == 1
        for a in iter_atoms(mol)
    ) or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != 2
            or len(inventory.principal) != 1 or inventory.principal[0].kind != 'acid'
            or sum(g.kind == 'carbamate' for g in inventory.groups) != 1):
        return None
    group = inventory.principal[0]
    alpha = mol.GetAtomWithIdx(group.owner)
    if (alpha.GetAtomicNum() != 6 or alpha.IsInRing() or alpha.GetIsAromatic()
            or alpha.GetTotalNumHs() != 2 or alpha.GetDegree() != 2
            or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1):
        return None
    roots = [n.GetIdx() for n in alpha.GetNeighbors() if n.GetIdx() != group.center]
    if len(roots) != 1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum() != 7:
        return None
    from .carbamate_components import mapped_alkoxycarbonylamino_proof

    root = roots[0]
    proof = mapped_alkoxycarbonylamino_proof(mol, Chem, root, group.owner)
    if (proof is None or set(proof.atoms) | set(group.atoms) | {group.owner} !=
            {a.GetIdx() for a in iter_atoms(mol)}):
        return None
    return _mapped_acetic_acid_parent(mol, Chem, group, root, proof.text, proof.atoms)


def _mapped_acetic_acid_parent(mol, Chem, group, root, text, owned):
    from .component_mapping import MappedComponent, decorate_component
    from .retained_acid_rules import retained_acid_locant_evidence

    oxygens = [i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
    if len(oxygens) != 2:
        return None
    double_oxygen = [i for i in oxygens
                     if mol.GetBondBetweenAtoms(group.center, i).GetBondTypeAsDouble() == 2]
    if len(double_oxygen) != 1:
        return None
    single_oxygen = next(i for i in oxygens if i != double_oxygen[0])
    mapping = ((group.center, '1'), (group.owner, '2'),
               (double_oxygen[0], "O'1"), (single_oxygen, 'O1'))
    parents = frozenset(dict(mapping))
    prefixes = (('2', text),)
    evidence = retained_acid_locant_evidence(
        mol, Chem, parent_class='acetic acid', principal_kind='acid',
        parent_atoms=parents, atom_locants=mapping, prefixes=prefixes,
    )
    if evidence is None:
        return None
    construction = MappedComponent(
        'acetic acid', mapping, parents, prefixes, frozenset(owned),
        retained_acid_locant_evidence=evidence,
        prefix_branch_ownership=((root, group.owner, frozenset(owned), text),),
    )
    name = decorate_component(mol, Chem, construction)
    return BoundedAcidParent(name, mapping, construction) if name else None


def aromatic_acetic_acid_parent(mol, Chem):
    """P-65.1.1.1 two-carbon acid with one mapped aromatic ring branch.

    Reject every other carbon-chain extension and every characteristic group
    except lower-priority ring lactams before selecting the retained parent.
    The mapped branch and final component writers separately check its exact
    ring-atom and carbonyl-oxygen ownership.
    """
    if mol.GetStereoGroups() or any(
        a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
        or int(a.GetChiralTag()) or a.GetAtomicNum() == 1
        for a in iter_atoms(mol)
    ) or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or any(g.kind not in {'acid', 'lactam'}
                                      for g in inventory.groups)
            or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'acid'):
        return None
    group = inventory.principal[0]
    alpha = mol.GetAtomWithIdx(group.owner)
    if (alpha.GetAtomicNum() != 6 or alpha.IsInRing() or alpha.GetIsAromatic()
            or alpha.GetTotalNumHs() != 2 or alpha.GetDegree() != 2
            or mol.GetBondBetweenAtoms(group.center, group.owner).GetBondTypeAsDouble() != 1):
        return None
    branches = [n.GetIdx() for n in alpha.GetNeighbors() if n.GetIdx() != group.center]
    if len(branches) != 1:
        return None
    root = branches[0]
    branch_atom = mol.GetAtomWithIdx(root)
    if (branch_atom.GetAtomicNum() != 6 or not branch_atom.IsInRing()
            or not branch_atom.GetIsAromatic()
            or mol.GetBondBetweenAtoms(root, group.owner).GetBondTypeAsDouble() != 1):
        return None
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms

    text, owned = mapped_branch(mol, Chem, root, group.owner)
    ring_atoms = {i for i in owned if mol.GetAtomWithIdx(i).IsInRing()}
    lactams = [g for g in inventory.groups if g.kind == 'lactam']
    if (not text or not owned or any(g.center not in ring_atoms
                                     or not set(g.atoms) <= set(owned)
                                     for g in lactams)
            or not ring_atoms <= mapped_branch_ring_atoms(
            mol, Chem, root, group.owner, text, owned)
            or set(owned) | set(group.atoms) | {group.owner} !=
            {a.GetIdx() for a in iter_atoms(mol)}):
        return None
    # The final retained-acid prefix writer verifies the admissible ring and
    # terminal substituent roles again on the source graph.
    from .complete_functional_parent import _render_functional_candidate

    return _render_functional_candidate(
        mol, Chem, (group,), 'acid', (group.center, group.owner), False, False,
        set(group.atoms) - {group.center}, [],
        mapped_ring_prefixes=True, charge_role_proofs=(),
    )


def substituted_benzoic_acid_parent(mol, Chem):
    """P-65.1.1.1 benzoic acid with bounded ring substituents.

    The sole carbonyl is attached directly to the only six-carbon aromatic
    ring. The complete fragment is this ring, its carboxyl group, and
    terminal halogen, methyl or source-recognized formyl groups. Thus no longer
    acyclic acid parent or second ring
    competes for the suffix. With the acid-bearing ring atom fixed at 1, P-14.4
    and P-14.5 choose between both ring orientations from graph locants.
    Annotations and other substituent classes require their own evidence.
    """
    if mol.GetStereoGroups() or any(
        a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
        or int(a.GetChiralTag()) or a.GetAtomicNum() == 1
        for a in iter_atoms(mol)
    ) or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)):
        return None
    inventory = selection_inventory(mol)
    if (not inventory.supported or any(g.kind not in {'acid', 'aldehyde'}
                                      for g in inventory.groups)
            or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'acid'):
        return None
    group = inventory.principal[0]
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return None
    ring = frozenset(rings[0])
    if (group.owner not in ring or any(
            mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
        return None
    ring_bonds = [bond for bond in iter_bonds(mol)
                  if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring]
    if len(ring_bonds) != 6 or any(not b.GetIsAromatic() for b in ring_bonds):
        return None
    if mol.GetBondBetweenAtoms(group.owner, group.center).GetBondTypeAsDouble() != 1:
        return None
    outside = {a.GetIdx() for a in iter_atoms(mol)} - ring - set(group.atoms)
    labels = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    prefixes = {}
    aldehyde_atoms = set()
    for aldehyde in (g for g in inventory.groups if g.kind == 'aldehyde'):
        carbon = mol.GetAtomWithIdx(aldehyde.center)
        if (aldehyde.owner not in ring or carbon.GetDegree() != 2
                or carbon.GetTotalNumHs() != 1 or len(aldehyde.atoms) != 2
                or mol.GetBondBetweenAtoms(aldehyde.owner, aldehyde.center).GetBondTypeAsDouble() != 1
                or set(aldehyde.atoms) & aldehyde_atoms
                or not set(aldehyde.atoms) <= outside):
            return None
        oxygens = [i for i in aldehyde.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
        if (len(oxygens) != 1 or mol.GetAtomWithIdx(oxygens[0]).GetDegree() != 1
                or mol.GetBondBetweenAtoms(aldehyde.center, oxygens[0]).GetBondTypeAsDouble() != 2):
            return None
        prefixes[aldehyde.center] = 'formyl'
        aldehyde_atoms.update(aldehyde.atoms)
    for idx in outside - aldehyde_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetIdx() not in ring
                or atom.GetNeighbors()[0].GetIdx() == group.owner
                or atom.GetBonds()[0].GetBondTypeAsDouble() != 1):
            return None
        if atom.GetAtomicNum() in labels:
            prefixes[idx] = labels[atom.GetAtomicNum()]
        elif atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 3:
            prefixes[idx] = 'methyl'
        else:
            return None
    if any(sum(n.GetIdx() in prefixes for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 1
           for i in ring):
        return None

    paths = substituted_benzene_paths(mol, ring, group.owner, prefixes)
    from .complete_functional_parent import _render_functional_candidate

    candidates = []
    for path in paths:
        candidate = _render_functional_candidate(
            mol, Chem, (group,), 'acid', path, True, True,
            set(group.atoms) - {group.center}, [],
            mapped_ring_prefixes=True, charge_role_proofs=(),
        )
        if candidate is None:
            return None
        candidates.append(candidate)
    if not candidates or len({candidate.name for candidate in candidates}) != 1:
        return None
    return candidates[0]
