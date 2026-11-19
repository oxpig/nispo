"""Fast-path checks for ambiguous names from the preserved public writer.

These checks use the source graph and the proposed name, not a name parser.
They are deliberately selective; passing them does not prove graph recovery.
"""

from collections import Counter
from functools import lru_cache
from importlib import resources
import json
import re

_STEREO_TOKEN = re.compile(r"(?<=[(,])(\d+[a-z]?'*)?([RSrsEZ])(?=[,)])")
_RELATIVE = re.compile(r'\((cis|trans)\)-')


@lru_cache(maxsize=1)
def _paired_standalone_aliases():
    """Parser aliases whose same stored graph also has an e-ending alias.

    Both variants can be legitimate: furan and coumarin are examples of
    correctly spelled names without a final e. These pairs are consequently
    a reason to ask the complete writer, never a rule for appending a vowel.
    """
    path = resources.files('nispo._fast_engine').joinpath(
        'resources', 'opsin_retained_mono_fused_ring_templates.json')
    rows = json.loads(path.read_text(encoding='utf-8'))
    aliases = {}
    for row in rows:
        if isinstance(row.get('name'), str) and isinstance(row.get('smiles'), str):
            aliases.setdefault(row['smiles'], set()).add(row['name'])
    return frozenset(name for names in aliases.values() for name in names
                     if name + 'e' in names)


@lru_cache(maxsize=1)
def _unfinished_fast_parent_forms():
    """Known parser aliases and contracted stems that lack a parent-hydride e.

    These strings can be valid *before a vowel-initial suffix*. They are not
    complete parents, including in a fusion name or before a suffix beginning
    with a consonant. The complete writer must decide the actual parent and
    indicated hydrogens; simply adding an e to the final text cannot do that.
    """
    from ._fast_engine.formatting import _PREFERRED_PARENT_STEMS
    from ._fast_engine.resource_templates import PARENT_NAME_NORMALIZATIONS

    forms = {stem for full, stem in _PREFERRED_PARENT_STEMS if full == stem + 'e'}
    forms.update(alias for alias, full in PARENT_NAME_NORMALIZATIONS.items()
                 if full == alias + 'e')
    # The compatibility monocycle writer also receives these exact parser
    # aliases; their full source parent forms end in ``-oline``.
    forms.update(('imidazolin', 'pyrazolin', 'pyrrolin', 'isooxazolin',
                  'isothiazolin', 'asym-triazin', 'sym-triazin'))
    stems = tuple(sorted(forms, key=lambda value: (-len(value), value)))
    # A consonant-starting suffix retains the e of the parent hydride (P-16.7).
    # At least one locant is required so ordinary substituent prefixes are not
    # mistaken for a suffix boundary.
    locanted_suffix = re.compile(
        r'(?:' + '|'.join(re.escape(stem) for stem in stems) + r')-'
        r"(?:\d+[a-z]?'*(?:,\d+[a-z]?'*)*)"
        r"(?:\(\d+[a-z]?'*H(?:,\d+[a-z]?'*H)*\))?-([a-z]+)"
    )
    return stems, locanted_suffix


def _unfinished_known_parent(name):
    stems, locanted_suffix = _unfinished_fast_parent_forms()
    if name.endswith(stems):
        return True
    return any(match.group(1)[0] not in 'aeiouy'
               for match in locanted_suffix.finditer(name))


def _unfinished_fast_parent(name):
    return _unfinished_known_parent(name) or name in _paired_standalone_aliases()


def _contains_carbamyl_alias(name):
    """P-34.2.1.1: carbamoyl is the complete prefix for NH2-C(O)-."""
    return 'carbamyl' in name


def _historical_partial_ring_parent(name):
    """Defer unqualified historical -oline parents to the mapped writer.

    The IUPAC recommendations replaced pyrroline, imidazoline and
    pyrazoline parents with appropriately hydrogenated mancude names. The
    old unlocanted words also do not distinguish all unsaturation isomers.
    """
    return name.endswith(('pyrroline', 'imidazoline', 'pyrazoline'))


def _ring_features(mol):
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if len(rings) < 2:
        return False, False, False

    spiro = any(len(ring & other) == 1 for i, ring in enumerate(rings)
                for other in rings[i + 1:])
    pending = list(rings)
    mixed_heterocycle = False
    fused_aromatic_nh = False
    while pending:
        core = set(pending.pop())
        count = 1
        while True:
            attached = [ring for ring in pending if ring & core]
            if not attached:
                break
            for ring in attached:
                pending.remove(ring)
                core.update(ring)
                count += 1
        if count < 2:
            continue
        atoms = [mol.GetAtomWithIdx(index) for index in core]
        hetero = any(atom.GetAtomicNum() != 6 for atom in atoms)
        mixed = (any(atom.GetIsAromatic() for atom in atoms)
                 and any(not atom.GetIsAromatic() for atom in atoms))
        mixed_heterocycle |= bool(hetero and mixed)
        fused_aromatic_nh |= any(
            atom.GetAtomicNum() == 7 and atom.GetIsAromatic()
            and atom.GetTotalNumHs(includeNeighbors=True) for atom in atoms)
    return spiro, mixed_heterocycle, fused_aromatic_nh


def _has_isolated_mancude_azole(mol):
    """Recognize the monocyclic parents with preferred 1,2/1,3 locants.

    The fast engine can emit an unlocanted oxazole/thiazole
    alias (including the nonpreferred isooxazol spelling). When a current
    compatibility candidate is unavailable, use the complete writer for
    these graph-proved isolated rings. Fused benzazole systems and other
    heteroatom counts are left to their existing routes.
    """
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    for i, ring in enumerate(rings):
        if len(ring) != 5 or any(ring & other for j, other in enumerate(rings) if i != j):
            continue
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if not all(atom.GetIsAromatic() for atom in atoms):
            continue
        numbers = Counter(atom.GetAtomicNum() for atom in atoms)
        if numbers == Counter({6: 3, 7: 1, 8: 1}) or numbers == Counter({6: 3, 7: 1, 16: 1}):
            return True
    return False


def _has_substituted_bridged_bicycle(mol):
    """Do not trust fast-engine substituent locants on a bridged bicycle.

    This is a graph-only necessary guard for the fast engine, not a
    preferred-name choice. Two cycles sharing a path of at least three atoms
    can have competing bridge numberings; a substituent needs the mapped
    numbering of the original core to retain its actual attachment site.
    """
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    for index, left in enumerate(rings):
        for right in rings[index + 1:]:
            if len(left & right) < 3:
                continue
            core = left | right
            edge_count = sum(bond.GetBeginAtomIdx() in core and bond.GetEndAtomIdx() in core
                             for bond in mol.GetBonds())
            if edge_count - len(core) + 1 != 2:
                continue
            if any(neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in core
                   for idx in core for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()):
                return True
    return False


def source_risk(mol, Chem):
    """Identify inputs better sent directly to the comprehensive writer."""
    atoms = tuple(mol.GetAtoms())
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(atom.GetAtomicNum() == 0 or atom.HasQuery()
                   or atom.GetAtomMapNum() or atom.GetIsotope()
                   or atom.GetNumRadicalElectrons() or atom.HasProp('_UnknownStereo')
                   for atom in atoms)
            or any(bond.HasQuery() or bond.GetBondDir() == Chem.BondDir.UNKNOWN
                   for bond in mol.GetBonds())):
        return True, False
    if any(atom.GetFormalCharge() for atom in atoms):
        from .neutral_charge_components import neutral_charge_role_proofs
        proofs = neutral_charge_role_proofs(mol)
        if not proofs or any(proof.kind not in {'nitro', 'azido'} for proof in proofs):
            return True, False
    spiro, mixed, fused_nh = _ring_features(mol)
    return (spiro or (fused_nh and mixed) or _has_isolated_mancude_azole(mol)
            or _has_substituted_bridged_bicycle(mol)), mixed


def _stereo_risk(mol, Chem, name):
    from .stereochemistry import (
        assigned_atom_stereochemistry, assigned_bond_stereochemistry,
        relative_monocycle_configuration,
    )

    atoms = [atom for atom in mol.GetAtoms() if int(atom.GetChiralTag())]
    bonds = [bond for bond in mol.GetBonds() if int(bond.GetStereo())]
    if not atoms and not bonds:
        return False
    if any(atom.GetAtomicNum() != 6 or atom.GetChiralTag() not in (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW) for atom in atoms):
        return True
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
    if (set(assigned) != {atom.GetIdx() for atom in atoms}
            or any(value not in {'R', 'S', 'r', 's'} for value in assigned.values())):
        return True
    tokens = [match.groups() for match in _STEREO_TOKEN.finditer(name)]
    atom_tokens = [(locant, value) for locant, value in tokens if value in 'RSrs']
    bond_tokens = [(locant, value) for locant, value in tokens if value in 'EZ']
    relative = _RELATIVE.findall(name)
    if relative:
        if (len(relative) != 1 or len(atoms) != 2 or atom_tokens
                or any(value not in {'r', 's'} for value in assigned.values())):
            return True
        tagged = set(assigned)
        rings = tuple(tuple(ring) for ring in mol.GetRingInfo().AtomRings())
        candidates = [ring for i, ring in enumerate(rings)
                      if tagged <= set(ring)
                      and not any(set(ring) & set(other)
                                  for j, other in enumerate(rings) if i != j)]
        if (len(candidates) != 1
                or relative_monocycle_configuration(mol, Chem, candidates[0]) != relative[0]):
            return True
    elif Counter(value for _, value in atom_tokens) != Counter(assigned.values()):
        return True
    if (len(atom_tokens) + len(bond_tokens) + len(relative) > 1
            and any(not locant for locant, _ in atom_tokens + bond_tokens)):
        return True
    actual = ({(first, second): value for first, second, value
               in assigned_bond_stereochemistry(mol, Chem)} if bonds else {})
    return (len(actual) != len(bonds)
            or Counter(value for _, value in bond_tokens) != Counter(actual.values()))


def name_risk(mol, Chem, name, mixed):
    """Screen the proposed spelling for cases needing the complete writer."""
    if (not name or name.startswith('(beta-allyl)-') or _contains_carbamyl_alias(name)
            or _historical_partial_ring_parent(name)
            or name.endswith(('en', 'yn')) or _unfinished_fast_parent(name)):
        return True
    if mixed and ('dihydro' in name or 'tetrahydro' in name or 'hydro-' in name):
        return True
    return _stereo_risk(mol, Chem, name)
