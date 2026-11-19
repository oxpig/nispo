"""Mapped acyclic pseudoketones derived from exocyclic N-acyl heterocycles.

P-64.3.2 explicitly makes the carbonyl component the parent for a hidden
amide. The acyl parent and the ring-N substituent are mapped separately.
"""

from collections import Counter
import re

from .component_mapping import mapped_branch
from .formatting import format_prefixes, join_prefix_parent
from .functional_inventory import functional_inventory
from .molecule_cache import mol_cache_get
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


RULE = "P-41/P-64.3.2/P-66.1.3"


def _has_complete_tetrahedral_descriptors(mol, Chem, prefix):
    """Require the mapped prefix to retain every explicitly assigned center.

    The producer does not admit double-bond stereo or isotope descriptors;
    stereocenters without a CIP descriptor are outside this bounded route.
    Unassigned stereogenic centers are never assigned by this check.
    """
    if mol.GetStereoGroups() or any(int(bond.GetStereo()) for bond in iter_bonds(mol)):
        return False
    work = Chem.Mol(mol)
    Chem.AssignStereochemistry(work, cleanIt=False, force=True)
    expected = []
    for atom in iter_atoms(work):
        if not int(atom.GetChiralTag()):
            continue
        if (atom.GetChiralTag() not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                       Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                or not atom.HasProp('_CIPCode')):
            return False
        code = atom.GetProp('_CIPCode')
        if code not in {'R', 'S', 'r', 's'}:
            return False
        expected.append(code)
    rendered = []
    for match in re.finditer(r'\((\d+[a-z]?[RSrs](?:,\d+[a-z]?[RSrs])*)\)', prefix):
        rendered.extend(re.findall(r'\d+[a-z]?([RSrs])', match.group(1)))
    return Counter(expected) == Counter(rendered)


def hidden_amide_pseudoketone_decision(mol, Chem):
    """Return a fully mapped acyl parent for a senior exocyclic carbonyl."""
    decision = _acetyl_decision(mol, Chem)
    if decision is not None:
        return decision
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'pseudoketone'):
        return None
    group = inventory.principal[0]
    sites = mol_cache_get(mol, 'nested_azido_sites')
    charge_proofs = tuple(sites.values()) if sites else ()
    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups() or group.ligand < 0
            or mol.GetAtomWithIdx(group.center).IsInRing()
            or any((a.GetFormalCharge() and not charge_proofs) or a.GetIsotope() or a.GetNumRadicalElectrons()
                   for a in iter_atoms(mol))):
        return None
    nitrogen = mol.GetAtomWithIdx(group.ligand)
    if nitrogen.GetAtomicNum() != 7 or not nitrogen.IsInRing():
        return None
    return _acyl_chain_decision(mol, Chem, group, charge_proofs=charge_proofs)


def _acyl_chain_decision(mol, Chem, group, *, charge_proofs=()):
    from .branch_rules import collect_subtree, parent_chain_name
    from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from .formatting import attach_suffix
    from .naming_result import SelectionLimitation
    from .parent_selection import graph_parent_facts, parent_frontier
    from .parent_rules import _ester_acyl_parent_paths

    # This producer covers ordinary carbon-rooted acyls. A ring-attached
    # carbonyl still belongs to an exocyclic ketone component, with the ring
    # substituent explicitly cited even when the acyl chain has one atom.
    if group.owner < 0:
        return None
    oxygen = next((i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8), None)
    if oxygen is None:
        return None
    if mol.GetAtomWithIdx(group.owner).IsInRing():
        paths = [(group.center,)]
    else:
        paths = [(group.center, *path) for path in
                 _ester_acyl_parent_paths(mol, Chem, group.center, group.owner)]
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    candidates = []
    for path in paths:
        mapping = {i: str(j) for j, i in enumerate(path, 1)}
        owners, prefixes = {oxygen}, []
        roles = []
        ownership = [] if charge_proofs else None
        for idx in path:
            for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                root = atom.GetIdx()
                if root in mapping or root in owners or atom.GetAtomicNum() == 1:
                    continue
                if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                    break
                text, atoms = mapped_general_branch(mol, Chem, root, idx)
                if not text and root == group.ligand and not charge_proofs:
                    from .polycyclic_rules import polycyclic_substituent_prefix
                    from .component_mapping import _general_branch_stereo_is_cited

                    atoms = frozenset(collect_subtree(mol, root, idx))
                    text = polycyclic_substituent_prefix(mol, Chem, root, idx, higher_bridges=True)
                    if not _general_branch_stereo_is_cited(mol, Chem, atoms, text):
                        text = ''
                if not text or atoms & (mapping.keys() | owners):
                    break
                prefixes.append((mapping[idx], text))
                if ownership is not None:
                    ownership.append((root, idx, atoms, text))
                roles.append(('substituent', atoms, ()))
                owners.update(atoms)
            else:
                continue
            break
        if set(path) | owners != heavy:
            continue
        base = parent_chain_name(mol, path)
        if not base:
            continue
        parent = attach_suffix(base, 'one', ('1',) if len(path) > 1 else ())
        component = MappedComponent(parent, tuple(mapping.items()), frozenset(path),
            tuple(prefixes), frozenset(owners), systematic_parent=True,
            omit_methyl_prefix_locant=len(path) == 1)
        name = decorate_component(mol, Chem, component)
        if not name:
            continue
        facts = graph_parent_facts(mol, path, mapping, principal_class=group.seniority,
            suffix_centers=(group.center,), suffix_locants=('1',), prefixes=prefixes, name=name,
            allow_annotations=True, charge_role_proofs=charge_proofs,
            prefix_branch_ownership=ownership or (), resolve_branch=mapped_general_branch)
        if charge_proofs and facts is None:
            continue
        candidates.append((name, component.atom_locants, facts,
                           (('carbonyl-parent', frozenset(path) | {oxygen}, component.atom_locants), *roles)))
    best = parent_frontier(candidates, lambda row: row[2])
    if len({row[0] for row in best}) != 1:
        return None
    name, mapping, _facts, roles = best[0]
    return MethodDecision(name, 'substitutive', frozenset({group.center}), heavy,
        RULE, mapping, component_roles=roles, preference_issues=(SelectionLimitation(
            'pseudoketone-parent', RULE, 'The acyl chain and the original ring-N branch are mapped.'),))


def _acetyl_decision(mol, Chem):
    if mol.GetNumHeavyAtoms() > 128:
        return None
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetIsotope()
           for atom in iter_atoms(mol)):
        return None
    inventory = functional_inventory(mol)
    if not inventory.supported or len(inventory.principal) != 1:
        return None
    if inventory.source_frontier.status == 'barrier':
        return None
    group = inventory.principal[0]
    if group.kind != 'pseudoketone':
        return None
    carbonyl = mol.GetAtomWithIdx(group.center)
    if carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing():
        return None
    neighbors = [n for n in carbonyl.GetNeighbors() if n.GetAtomicNum() > 1]
    oxygens = [n for n in neighbors if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 2
               and n.GetDegree() == 1]
    nitrogens = [n for n in neighbors if n.GetAtomicNum() == 7 and n.IsInRing()
                 and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 1]
    methyls = [n for n in neighbors if n.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 1
               and n.GetTotalNumHs(includeNeighbors=True) == 3
               and sum(a.GetAtomicNum() > 1 for a in n.GetNeighbors()) == 1]
    if len(neighbors) != 3 or len(oxygens) != 1 or len(nitrogens) != 1 or len(methyls) != 1:
        return None
    parent_atoms = frozenset((group.center, oxygens[0].GetIdx(), methyls[0].GetIdx()))
    prefix, branch_atoms = mapped_branch(mol, Chem, nitrogens[0].GetIdx(), group.center)
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if (not prefix or not branch_atoms or parent_atoms & branch_atoms
            or parent_atoms | branch_atoms != heavy
            or not _has_complete_tetrahedral_descriptors(mol, Chem, prefix)):
        return None
    name = join_prefix_parent(
        format_prefixes([('1', prefix)], hierarchical_enclosures=True), 'ethan-1-one')
    if not name:
        return None
    roles = (
        ('carbonyl_parent', parent_atoms,
         ((group.center, '1'), (methyls[0].GetIdx(), '2'))),
        ('heterocyclic_nitrogen_component', branch_atoms, ()),
    )
    return MethodDecision(name, 'substitutive', frozenset({group.center}), heavy,
                          RULE, ((group.center, '1'),), component_roles=roles)
