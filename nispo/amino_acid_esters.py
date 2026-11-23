"""Retained amino-acid words for a selected, source-mapped ester."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _residue_queries(Chem):
    alpha = '[C;X4;H1;!R]([N;X3;+0;!R])'
    patterns = (
        ('alaninate', alpha + '[CH3]'),
        ('valinate', alpha + '[CH]([CH3])[CH3]'),
        ('phenylalaninate', alpha + '[CH2][c]1[cH][cH][cH][cH][cH]1'),
        ('prolinate', '[C;X4;H1]1[N;X3;+0][CH2][CH2][CH2]1'),
    )
    return tuple((name, Chem.MolFromSmarts(pattern)) for name, pattern in patterns)


@lru_cache(maxsize=1)
def _amino_ester_query(Chem):
    return Chem.MolFromSmarts('[C;X4;H1]([N;X3;+0])[CX3](=[OX1])[OX2][#6]')


def retained_amino_acid_ester_decision(mol, Chem):
    """Name a single selected amino-acid ester before generic parent search."""
    if not any(mol.GetAtomWithIdx(site[0]).GetChiralTag()
               for site in mol.GetSubstructMatches(_amino_ester_query(Chem))):
        return None
    if mol.GetStereoGroups() or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery()
                                   for a in mol.GetAtoms()):
        return None
    from .extended_inventory import selection_inventory
    from .mapped_ester_general import mapped_ester_arm
    from .mapped_monoester_general import _source_ester
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    inventory = selection_inventory(mol)
    group = _source_ester(inventory)
    if group is None or not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs):
        return None
    arm = mapped_ester_arm(mol, Chem, group)
    if arm is None:
        return None
    name = retained_amino_acid_ester(mol, Chem, group, arm.name, arm.atoms)
    if not name:
        return None
    rule = 'P-41/P-65.6/P-103.2.3/P-103.2.6'
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    organyl = arm.atoms | {group.ligand}
    return MethodDecision(name, 'functional-class', frozenset({group.center}), scope, rule,
                          component_roles=(('acid', scope - organyl, ()), ('O-organyl', organyl, ())),
                          preference_issues=(SelectionLimitation(
                              'ester-parent', rule,
                              'The retained amino-acid residue and both ester arms are mapped.'),))


def retained_amino_acid_ester(mol, Chem, group, organyl, organyl_atoms):
    """Apply P-103.2.3 and P-103.2.6 to an ordinary L/D residue."""
    if group.kind != 'ester' or group.owner < 0:
        return ''
    alpha = mol.GetAtomWithIdx(group.owner)
    if (not alpha.GetChiralTag() or alpha.GetAtomicNum() != 6
            or not any(a.GetAtomicNum() == 7 for a in alpha.GetNeighbors())):
        return ''

    from .component_mapping import mapped_general_branch
    from .formatting import format_prefixes
    from .stereochemistry import assigned_atom_stereochemistry

    descriptor = dict(assigned_atom_stereochemistry(mol, Chem)).get(group.owner)
    if descriptor not in {'R', 'S'}:
        return ''
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    for parent, query in _residue_queries(Chem):
        for match in mol.GetSubstructMatches(query):
            if match[0] != group.owner:
                continue
            residue = frozenset(match)
            nitrogen = mol.GetAtomWithIdx(match[1])
            if residue & (frozenset(group.atoms) | organyl_atoms):
                continue
            if any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in (mol.GetAtomWithIdx(i) for i in residue)):
                return ''
            # The alpha configurations of these four mapped skeletons have
            # the same L/S correspondence, even after N substitution.
            config = 'L' if descriptor == 'S' else 'D'
            occupied = residue | frozenset(group.atoms) | organyl_atoms
            prefixes = []
            for child in nitrogen.GetNeighbors():
                root = child.GetIdx()
                if root in residue or child.GetAtomicNum() == 1:
                    continue
                word, owned = mapped_general_branch(mol, Chem, root, nitrogen.GetIdx())
                if not word or any(ch.isspace() for ch in word) or owned & occupied:
                    return ''
                occupied |= owned
                prefixes.append(('N', word))
            if occupied != scope:
                return ''
            prefix = format_prefixes(prefixes, hierarchical_enclosures=True).rstrip('-')
            return organyl + ' ' + (prefix + '-' if prefix else '') + config + '-' + parent
    return ''
