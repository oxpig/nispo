"""P-66.1.5.2.1 heterones on numbered benzene-fused sultam parents."""

from .component_mapping import mapped_branch
from .constants import MULT
from .formatting import (
    attach_suffix,
    format_prefixes,
    join_prefix_parent,
    multiplied_suffix,
    prefix_alphabetization_key,
)
from .functional_inventory import functional_inventory
from .hydrogen_accommodation import annular_pi_bonds, joint_hydrogen_states, peripheral_ring_atoms
from .mapped_ring_parents import ring_systems, skeletons
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .parent_selection import locant_key
from .ring_state_parent import _CARBONYLS, _ring_numbering_terms, _skeleton_references


RULE = "P-41/P-64.4/P-66.1.5.2.1"


def _ring_carbonyls(mol, Chem, sulfur, oxygens, core):
    """Prove the other class-16 suffix sites on the same ring system."""
    # The ordinary inventory needs a recognition copy without the annular
    # S(VI) site. Translate its carbonyl roles back to the original graph.
    graph = Chem.RWMol(mol)
    substitute = graph.GetAtomWithIdx(sulfur)
    substitute.SetAtomicNum(6)
    substitute.SetNumExplicitHs(0)
    substitute.SetNoImplicit(False)
    original = tuple(idx for idx in range(mol.GetNumAtoms()) if idx not in oxygens)
    for index in sorted(oxygens, reverse=True):
        graph.RemoveAtom(index)
    proxy = graph.GetMol()
    try:
        Chem.SanitizeMol(proxy)
    except (ValueError, RuntimeError):
        return None
    inventory = functional_inventory(proxy)
    if not inventory.supported:
        return None
    carbonyls = {}
    for group in inventory.groups:
        if group.seniority[0] > 16:
            continue
        center = original[group.center]
        if group.kind not in _CARBONYLS or center not in core:
            return None
        atom = mol.GetAtomWithIdx(center)
        oxygen = [
            neighbor.GetIdx() for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 8 and neighbor.GetDegree() == 1
            and mol.GetBondBetweenAtoms(center, neighbor.GetIdx()).GetBondTypeAsDouble() == 2
        ]
        if len(oxygen) != 1:
            return None
        carbonyls[center] = oxygen[0]
    return carbonyls


def _benzo_sultam_core(mol, sulfur, nitrogen):
    """Admit one benzenoid face and one five- to seven-membered sultam face."""
    core = next((core for core in ring_systems(mol) if sulfur in core), ())
    if not 9 <= len(core) <= 11:
        return None
    rings = [frozenset(ring) for ring in mol.GetRingInfo().AtomRings() if set(ring) <= core]
    if len(rings) != 2 or len(rings[0] & rings[1]) != 2:
        return None
    sultam, benzene = sorted(rings, key=lambda ring: sulfur not in ring)
    if (nitrogen not in sultam or sulfur in benzene or len(benzene) != 6
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in benzene)):
        return None
    return core


def _sultam_hydride_names(mol, Chem, frame, sulfur, oxygens, carbonyls):
    """Use the shared P-58 comparison with both S=O sites reserved."""
    parent, sequence, locants, elements, edges, _, fixed_key = frame[:7]
    mapping = dict(zip(sequence, locants, strict=True))
    sulfur_locant = mapping[sulfur]
    hetero_locants, _, stem = parent.partition('-')
    locant_words = hetero_locants.split(',')
    if sulfur_locant not in locant_words:
        return
    locant_words[locant_words.index(sulfur_locant)] += 'lambda6'
    hydride = ','.join(locant_words) + '-' + stem
    skeleton_and_sulfur_oxygen = frozenset(sequence) | oxygens
    capped = []
    for idx in sequence:
        atom = mol.GetAtomWithIdx(idx)
        external_order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds()
                             if bond.GetOtherAtomIdx(idx) not in skeleton_and_sulfur_oxygen
                             and bond.GetOtherAtom(atom).GetAtomicNum() > 1)
        capped.append(atom.GetTotalNumHs(includeNeighbors=True) + int(external_order))
    actual = tuple(capped)
    actual_pi = annular_pi_bonds(mol, Chem, sequence)
    demands = tuple(2 if idx in carbonyls else 0 for idx in sequence)
    sites = (sulfur_locant, sulfur_locant, *(mapping[center] for center in carbonyls))
    suffix = multiplied_suffix(len(sites), 'one')
    if not suffix:
        return
    perimeter = peripheral_ring_atoms(edges)
    for reference_h, indicated, _ in _skeleton_references(frame):
        states = joint_hydrogen_states(elements, edges, reference_h, actual, demands, actual_pi)
        terms = _ring_numbering_terms(locants, edges, perimeter, fixed_key, indicated,
                                     sites, states, None, False)
        for _, added, hydro, leading, suffix_locants, key in terms:
            leading_text = ','.join(locant + 'H' for locant in leading) + '-' if leading else ''
            suffix_position = ','.join(suffix_locants)
            if added:
                suffix_position += '(' + ','.join(locant + 'H' for locant in added) + ')'
            name = attach_suffix(leading_text + hydride, suffix, (suffix_position,))
            if hydro:
                name = ','.join(hydro) + '-' + MULT[len(hydro)] + 'hydro-' + name
            yield key, name, mapping


def cyclic_sultam_heterone_decision(mol, Chem):
    """Name a fully mapped benzene-fused sultam as a λ⁶ heterone."""
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    sulfur_atoms = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 16]
    if len(sulfur_atoms) != 1 or not sulfur_atoms[0].IsInRing():
        return None
    sulfur = sulfur_atoms[0]
    sidx = sulfur.GetIdx()
    neighbors = [n for n in sulfur.GetNeighbors() if n.GetAtomicNum() > 1]
    oxygens = frozenset(n.GetIdx() for n in neighbors
                        if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                        and mol.GetBondBetweenAtoms(sidx, n.GetIdx()).GetBondTypeAsDouble() == 2)
    ring_neighbors = [n for n in neighbors if n.IsInRing()
                      and mol.GetBondBetweenAtoms(sidx, n.GetIdx()).GetBondTypeAsDouble() == 1]
    nitrogens = [n for n in ring_neighbors if n.GetAtomicNum() == 7]
    if (len(neighbors) != 4 or len(oxygens) != 2 or len(ring_neighbors) != 2
            or len(nitrogens) != 1 or sulfur.GetFormalCharge() or sulfur.GetTotalValence() != 6):
        return None
    core = _benzo_sultam_core(mol, sidx, nitrogens[0].GetIdx())
    if core is None or any(n.GetIdx() not in core for n in ring_neighbors):
        return None
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() or int(a.GetChiralTag())
           or a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53} for a in iter_atoms(mol)):
        return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return None
    if any(a.GetFormalCharge() for a in iter_atoms(mol)):
        from .neutral_charge_components import neutral_charge_role_proofs, validate_neutral_charge_role_proofs

        if not validate_neutral_charge_role_proofs(mol, neutral_charge_role_proofs(mol)):
            return None
    carbonyls = _ring_carbonyls(mol, Chem, sidx, oxygens, core)
    if carbonyls is None:
        return None
    suffix_atoms = oxygens | frozenset(carbonyls.values())
    owned = set(core | suffix_atoms)
    branches = []
    for idx in sorted(core):
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            other = neighbor.GetIdx()
            if other in core or other in suffix_atoms or neighbor.GetAtomicNum() == 1:
                continue
            if bond.GetBondTypeAsDouble() != 1:
                return None
            prefix, atoms = mapped_branch(mol, Chem, other, idx)
            if not prefix or not atoms or owned & atoms:
                return None
            owned.update(atoms)
            branches.append((idx, prefix, atoms))
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if owned != heavy:
        return None
    choices = []
    for frame in skeletons(mol, Chem, core):
        if len(frame) != 8 or frame[7] != 'benzo-heterocycle':
            continue
        for key, body, mapping in _sultam_hydride_names(mol, Chem, frame, sidx, oxygens, carbonyls):
            prefixes = [(mapping[idx], prefix) for idx, prefix, _ in branches]
            locant_key_set = tuple(sorted(locant_key(locant) for locant, _ in prefixes))
            citation_key = tuple(locant_key(locant) for locant, _ in sorted(
                prefixes, key=lambda row: (prefix_alphabetization_key(row[1]), locant_key(row[0]))))
            name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), body)
            choices.append(((key, locant_key_set, citation_key), name, mapping))
    if not choices:
        return None
    key = min(row[0] for row in choices)
    finalists = [(name, mapping) for rank, name, mapping in choices if rank == key]
    if len({name for name, _ in finalists}) != 1:
        return None
    name, mapping = finalists[0]
    locants = tuple(sorted(mapping.items()))
    roles = [('heterocyclic_parent', core, locants), ('oxo_suffixes', suffix_atoms, ())]
    roles.extend(('substituent:' + mapping[idx], atoms, ()) for idx, _, atoms in branches)
    unspecified = tuple(site for site in Chem.FindPotentialStereo(mol)
                        if str(site.specified) != 'Specified')
    issues = ((SelectionLimitation(
        'stereochemistry', 'P-91.1/P-93.1.1',
        f'The input leaves {len(unspecified)} potentially stereogenic unit(s) unspecified; '
        'the name does not assign their configurations.'),) if unspecified else ())
    return MethodDecision(name, 'substitutive', frozenset({sidx, *carbonyls}), heavy, RULE,
                          locants, component_roles=tuple(roles), preference_issues=issues)
