"""Bounded carboxamide suffixes on a source-mapped aromatic azole N.

P-66.1.1.1.1.3 uses a carboxamide suffix when the characteristic group is
attached to a ring atom. A private capped parent hydride proves the original
pyrrolic N hydrogen state and ring numbering; every emitted suffix and prefix
is then reconstructed and owned on the unmodified input graph.
"""
from dataclasses import dataclass

from .branch_rules import collect_subtree
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .input_stereochemistry import unsupported_relative_stereochemistry
from .mapped_ring_parents import capped_subtree_fragment, ring_systems
from .naming_context import MoleculeContext
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .ring_state_parent import select_ring_state


@dataclass(frozen=True, slots=True)
class _Site:
    center: int
    host: int
    oxygen: int
    ligand: int
    n_roots: tuple[int, ...]


def _carbon_tree(mol, root, blocked):
    """Prove one saturated carbon-only tree and its single original cut."""
    atoms = frozenset(collect_subtree(mol, root, blocked))
    if (not atoms or root not in atoms or blocked in atoms
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).IsInRing()
                   or mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)):
        return None
    cuts = {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
    bonds = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    if (cuts != {(root, blocked)} or len(bonds) != len(atoms) - 1
            or any(b.GetBondTypeAsDouble() != 1 for b in bonds)
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
        return None
    return atoms


def _amide_site(mol, ring):
    centers = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 3:
            continue
        carbonyl = [n.GetIdx() for n in atom.GetNeighbors()
                    if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                    and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
        ns = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 7
              and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(carbonyl) != 1 or len(ns) != 2:
            continue
        hosts = [i for i in ns if i in ring]
        ligands = [i for i in ns if i not in ring]
        if len(hosts) != 1 or len(ligands) != 1:
            continue
        host, ligand = hosts[0], ligands[0]
        host_atom, ligand_atom = mol.GetAtomWithIdx(host), mol.GetAtomWithIdx(ligand)
        roots = tuple(n.GetIdx() for n in ligand_atom.GetNeighbors() if n.GetIdx() != atom.GetIdx())
        if (not host_atom.GetIsAromatic() or host_atom.GetDegree() != 3
                or host_atom.GetTotalValence() != 3 or host_atom.GetTotalNumHs() != 0
                or ligand_atom.IsInRing() or ligand_atom.GetTotalValence() != 3
                or len(roots) > 2 or ligand_atom.GetTotalNumHs() != 2 - len(roots)
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots)):
            continue
        centers.append(_Site(atom.GetIdx(), host, carbonyl[0], ligand, roots))
    return centers[0] if len(centers) == 1 else None


def aromatic_n_carboxamide_decision(mol, Chem):
    """Construct a neutral imidazole/pyrazole N-carboxamide or decline.

    One ring, one terminal carboxamide and only carbon-tree, halo or bounded
    alkylsulfanyl branches close the whole source-class frontier locally.
    Unsupported annotations, additional functions and unproved N states
    decline. The complete preferred-name comparison remains unresolved.
    """
    atoms = tuple(mol.GetAtoms())
    if (mol.GetNumHeavyAtoms() > 64 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups() or unsupported_relative_stereochemistry(mol, Chem)
            or len(mol.GetRingInfo().AtomRings()) != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
                   or a.HasQuery() or a.GetAtomMapNum() or int(a.GetChiralTag())
                   for a in atoms)
            or any(b.HasQuery() or int(b.GetStereo()) or int(b.GetBondDir())
                   for b in mol.GetBonds())):
        return None
    rings = ring_systems(mol)
    if len(rings) != 1:
        return None
    ring = rings[0]
    if (len(ring) != 5 or sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring)
            != [6, 6, 6, 7, 7] or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)):
        return None
    site = _amide_site(mol, ring)
    if site is None:
        return None
    other_nitrogens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                       and i != site.host]
    if (len(other_nitrogens) != 1
            or mol.GetAtomWithIdx(other_nitrogens[0]).GetDegree() != 2
            or mol.GetAtomWithIdx(other_nitrogens[0]).GetTotalNumHs() != 0
            or {a.GetIdx() for a in atoms if a.GetAtomicNum() == 8} != {site.oxygen}
            or {a.GetIdx() for a in atoms if a.GetAtomicNum() == 7} !=
                {site.host, other_nitrogens[0], site.ligand}):
        return None
    # Only the original aromatic ring and the unique amide carbonyl may be
    # unsaturated; this rules out additional acid/ester/nitrile/oxo classes.
    if any(not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() != 1
           and {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} != {site.center, site.oxygen}
           for bond in mol.GetBonds()):
        return None
    if Chem.FindPotentialStereo(mol):
        return None

    # The cap replaces the removed external N-C bond by a parent-hydride N-H.
    # Aromatic N valence, source bond topology and cap sanitization are all
    # checked by the helper before its numbering/state is consumed here.
    capped = capped_subtree_fragment(mol, Chem, ring)
    if capped is None:
        return None
    fragment, original_to_local = capped
    if (set(original_to_local) != set(ring) or fragment.GetNumAtoms() != len(ring)
            or len(set(original_to_local.values())) != len(ring)):
        return None
    for original, local in original_to_local.items():
        source_atom = mol.GetAtomWithIdx(original)
        capped_atom = fragment.GetAtomWithIdx(local)
        if (source_atom.GetAtomicNum() != capped_atom.GetAtomicNum()
                or source_atom.GetFormalCharge() != capped_atom.GetFormalCharge()
                or source_atom.GetIsAromatic() != capped_atom.GetIsAromatic()):
            return None
    for i in ring:
        for j in ring:
            if i >= j:
                continue
            source_bond = mol.GetBondBetweenAtoms(i, j)
            capped_bond = fragment.GetBondBetweenAtoms(original_to_local[i], original_to_local[j])
            if ((source_bond is None) != (capped_bond is None)
                    or source_bond is not None and
                    (source_bond.GetBondType() != capped_bond.GetBondType()
                     or source_bond.GetIsAromatic() != capped_bond.GetIsAromatic())):
                return None
    capped_host = fragment.GetAtomWithIdx(original_to_local[site.host])
    if (not capped_host.GetIsAromatic() or capped_host.GetTotalNumHs() != 1
            or capped_host.GetDegree() != 2):
        return None
    state = select_ring_state(MoleculeContext(fragment, Chem))
    if state is None or len(state.candidates) != 1:
        return None
    record = state.selected
    if (record.parent not in {'imidazole', 'pyrazole'} or not record.hydrogen_budget_proved
            or not record.annotation_coverage or record.preference_issues
            or record.indicated_h_locants != ('1',)):
        return None
    reverse = {local: original for original, local in original_to_local.items()}
    mapping = {reverse[i]: loc for i, loc in record.atom_locants}
    expected_other_n = '3' if record.parent == 'imidazole' else '2'
    if (set(mapping) != set(ring) or mapping[site.host] != '1'
            or mapping[other_nitrogens[0]] != expected_other_n):
        return None

    suffix = frozenset({site.center, site.oxygen, site.ligand})
    prefixes, branches, ownership, roles = [], set(), [], []
    compound_names = set()
    for root in site.n_roots:
        owned = _carbon_tree(mol, root, site.ligand)
        text, mapped = mapped_branch(mol, Chem, root, site.ligand)
        if owned is None or not text or mapped != owned or branches & owned:
            return None
        prefixes.append(('N', text))
        branches.update(owned)
        ownership.append((root, site.ligand, owned, text))
        roles.append(('amide-N-substituent', owned, ()))
    for host in ring:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbor.GetIdx()
            if root in ring or root == site.center:
                continue
            if mol.GetAtomWithIdx(host).GetAtomicNum() != 6:
                return None
            if neighbor.GetAtomicNum() == 6:
                owned = _carbon_tree(mol, root, host)
            elif neighbor.GetAtomicNum() in {9, 17, 35, 53}:
                owned = (frozenset({root}) if neighbor.GetDegree() == 1
                         and mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() == 1
                         else None)
            elif neighbor.GetAtomicNum() == 16:
                children = [n.GetIdx() for n in neighbor.GetNeighbors() if n.GetIdx() != host]
                if neighbor.GetDegree() != 2 or len(children) != 1:
                    return None
                carbon_arm = _carbon_tree(mol, children[0], root)
                carbon_name, carbon_owned = mapped_branch(mol, Chem, children[0], root)
                if (carbon_arm is None or carbon_owned != carbon_arm
                        or carbon_name not in {'methyl', 'ethyl', 'propyl', 'butyl'}):
                    return None
                owned = carbon_arm | {root}
            else:
                return None
            text, mapped = mapped_branch(mol, Chem, root, host)
            if (owned is None or not text or mapped != owned or branches & owned
                    or owned & (ring | suffix)):
                return None
            if neighbor.GetAtomicNum() == 16:
                if text != carbon_name + 'sulfanyl':
                    return None
                compound_names.add(text)
            prefixes.append((mapping[host], text))
            branches.update(owned)
            ownership.append((root, host, owned, text))
            roles.append(('ring-substituent', owned, ()))
    heavy = frozenset(a.GetIdx() for a in atoms)
    if ring & suffix or branches & (ring | suffix) or ring | suffix | branches != heavy:
        return None
    atom_locants = tuple(mapping.items()) + (
        (site.center, ''), (site.oxygen, ''), (site.ligand, 'N'))
    component = MappedComponent(
        parent=record.name + '-1-carboxamide', atoms=ring | suffix,
        atom_locants=atom_locants, prefixes=tuple(prefixes),
        embedded_atoms=frozenset(branches), systematic_parent=True,
        nitrogen_site_locants=True, prefix_branch_ownership=tuple(ownership),
        compound_prefix_names=tuple(sorted(compound_names)),
    )
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    return MethodDecision(
        name, 'substitutive', frozenset({site.center}), heavy,
        'P-41/P-14.7.1/P-58.2.1/P-66.1.1.1.1.3/P-66.1.1.3.1.1',
        atom_locants, component_roles=(('parent-hydride', ring, tuple(mapping.items())),
            ('carboxamide', suffix, atom_locants[-3:]), *roles),
        preference_issues=(SelectionLimitation(
            'aromatic-N-carboxamide-parent', 'P-41/P-44/P-45/P-66.1.1',
            'The source-mapped ring and principal carboxamide are available; complete preferred-parent comparison remains unresolved.'),),
    )
