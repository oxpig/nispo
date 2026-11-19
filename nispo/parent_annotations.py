"""Opt-in P-44/P-82 evidence from original atom-to-locant bindings.

Parent choice and equivalent numbering are distinct operations. The caller
supplies the complete parent/suffix map, excluding separately named branches.
Advanced or unspecified configuration comparisons remain unknown.
"""
from collections import Counter
from functools import lru_cache

from .graph_model import graph_view, mapped_graph_components
from . import stereochemistry as _stereochemistry


@lru_cache(maxsize=1)
def _comparison_module():
    # Resolve the mutual annotation/parent dependency only after imports finish.
    from . import parent_selection
    return parent_selection


@lru_cache(maxsize=1)
def _stereo_runtime():
    # Keep RDKit optional at package-import time and retain live module helpers.
    from rdkit import Chem
    return Chem, _stereochemistry


def isotope_numbering_key(mol, mapping, *, include_substituent_groups=False, locant_series=None):
    """P-82.5.2/P-14.4(i) for equivalent numberings of one fixed parent.

    Chemical criteria must already have tied. Unlike parent-selection isotope
    evidence, this key first compares the complete, repeated series of modified
    positions. Nuclide seniority breaks only a tie in that series. If requested,
    each separately owned, singly attached labelled branch contributes one
    modified-group position; its internal isotope atoms do not multiply that
    parent attachment locant. Callers must separately prove the branch wording.
    """
    locant_set = _comparison_module().locant_set
    if locant_series is not None:
        # A typed parent producer may supply its source-bounded composite
        # locant order. Legacy callers retain the partial comparator above.
        locant_set = locant_series

    view = graph_view(mol, None)
    if not any(view.isotopes):
        return (), ()
    positions = []
    species = {}
    for idx, locant in mapping.items():
        if view.isotopes[idx]:
            positions.append(locant)
            nuclide = view.atomic_nums[idx], view.isotopes[idx]
            species.setdefault(nuclide, []).append(locant)
    if include_substituent_groups:
        for component, attachments in mapped_graph_components(
                mol, mapping, include_isotopic_hydrogens=True):
            nuclides = {(view.atomic_nums[i], view.isotopes[i]) for i in component
                        if view.isotopes[i]}
            if nuclides:
                anchors = frozenset(attachments)
                if len(anchors) != 1:
                    return None
                locant = mapping[next(iter(anchors))]
                positions.append(locant)
                for nuclide in nuclides:
                    species.setdefault(nuclide, []).append(locant)
    combined = locant_set(positions)
    if combined is None:
        return None
    ordered_species = []
    for (number, mass), values in sorted(species.items(), reverse=True):
        locants = locant_set(values)
        if locants is None:
            return None
        ordered_species.append((number, mass, locants))
    return combined, tuple(ordered_species)


def isotope_seniority(mol, mapping):
    locant_set = _comparison_module().locant_set

    view = graph_view(mol, None)
    if not any(view.isotopes):
        return (0, (), (), (), (), ())
    labels = [(i, view.atomic_nums[i], view.isotopes[i]) for i in mapping if view.isotopes[i]]
    if not labels:
        return (0, (), (), (), (), ())
    locations = locant_set(mapping[i] for i, _, _ in labels)
    if locations is None:
        return None
    numbers = sorted({number for _, number, _ in labels}, reverse=True)
    nuclides = sorted({(number, mass) for _, number, mass in labels}, reverse=True)
    counts = Counter(number for _, number, _ in labels)
    masses = Counter((number, mass) for _, number, mass in labels)
    return (-len(labels), tuple((-z, -counts[z]) for z in numbers),
            tuple((-z, -mass, -masses[z, mass]) for z, mass in nuclides),
            locations,
            tuple((z, locant_set(mapping[i] for i, number, _ in labels if number == z))
                  for z in numbers),
            tuple((z, mass, locant_set(mapping[i] for i, number, isotope in labels
                                     if (number, isotope) == (z, mass)))
                  for z, mass in nuclides))


def substituent_isotopes(mol, mapping):
    """P-45.4: attachment locants of separately owned labelled branches."""
    locant_set = _comparison_module().locant_set

    view = graph_view(mol, None)
    if not any(view.isotopes):
        return ()
    groups = []
    for component, attachments in mapped_graph_components(
            mol, mapping, include_isotopic_hydrogens=True):
        labels = tuple((view.atomic_nums[i], view.isotopes[i]) for i in component
                       if view.isotopes[i])
        if labels:
            anchors = frozenset(attachments)
            if len(anchors) != 1:
                return None
            groups.append((mapping[next(iter(anchors))], labels))
    if not groups:
        return ()
    locants = locant_set(loc for loc, _ in groups)
    if locants is None:
        return None
    numbers = sorted({number for _, labels in groups for number, _ in labels}, reverse=True)
    nuclides = sorted({nuclide for _, labels in groups for nuclide in labels}, reverse=True)
    return (locants,
            tuple((z, locant_set(loc for loc, labels in groups for number, _ in labels
                                if number == z)) for z in numbers),
            tuple((z, mass, locant_set(loc for loc, labels in groups for nuclide in labels
                                      if nuclide == (z, mass)))
                  for z, mass in nuclides))


def stereo_seniority(mol, mapping):
    """Bounded P-44.4.1.12; at most two ordinary assigned C centers.

    The feature-count signature is not a priority. Comparisons with different
    signatures must remain unknown, as must unspecified/non-tetrahedral units.
    """
    Chem, stereo = _stereo_runtime()
    comparison = _comparison_module()
    locant_key, locant_set = comparison.locant_key, comparison.locant_set

    potential = stereo._potential_stereochemistry(mol, Chem)
    assigned = dict(stereo.assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    bonds = stereo._indexed_bond_stereochemistry(mol, Chem)
    atoms = []
    for feature in potential:
        if feature.type == Chem.StereoType.Atom_Tetrahedral:
            idx = feature.centeredOn
            if idx not in mapping:
                continue
            if idx not in assigned or assigned[idx] not in {'R', 'S', 'r', 's'}:
                return None
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
                return None
            atoms.append((mapping[idx], assigned[idx]))
        elif feature.type == Chem.StereoType.Bond_Double:
            bond = mol.GetBondWithIdx(feature.centeredOn)
            ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if ends & mapping.keys() and (not ends <= mapping.keys() or not any(
                    {a, b} == ends for a, b, _ in bonds)):
                return None
        elif feature.type != Chem.StereoType.Bond_Double:
            return None
    mapped_bonds = [(a, b, d) for a, b, d in bonds if {a, b} <= mapping.keys()]
    if not atoms and not mapped_bonds:
        return ()
    if len(atoms) > 2 or any(locant_key(loc) is None for loc, _ in atoms):
        return None
    descriptors = tuple(d for _, d in sorted(atoms, key=lambda x: locant_key(x[0])))
    if len(descriptors) == 2 and any(d not in {'R', 'S'} for d in descriptors):
        return None
    like = (0 if descriptors[0] == descriptors[1] else 1) if len(descriptors) == 2 else 0
    descriptor_rank = tuple({'r': 0, 's': 1, 'R': 2, 'S': 3}[d] for d in descriptors)
    z_locs = locant_set(min((mapping[a], mapping[b]), key=locant_key)
                       for a, b, d in mapped_bonds if d == 'Z')
    if z_locs is None:
        return None
    return ((len(mapped_bonds), len(atoms)),
            (-sum(d == 'Z' for _, _, d in mapped_bonds), z_locs, like, descriptor_rank))


def stereo_identity(mol, mapping):
    """Same original units at the same locants prove configuration evidence ties.

    This also works for unspecified configurations: it never guesses what an
    unspecified configuration is or compares it with a different physical unit.
    """
    Chem, stereo = _stereo_runtime()

    assigned = dict(stereo.assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    records = []
    for feature in stereo._potential_stereochemistry(mol, Chem):
        if feature.type == Chem.StereoType.Atom_Tetrahedral:
            idx = feature.centeredOn
            if idx in mapping:
                records.append(('atom', idx, str(mapping[idx]), assigned.get(idx, '?')))
        elif feature.type == Chem.StereoType.Bond_Double:
            bond = mol.GetBondWithIdx(feature.centeredOn)
            ends = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            if ends & mapping.keys():
                if not ends <= mapping.keys():
                    return None
                records.append(('bond', bond.GetIdx(),
                                tuple(sorted((idx, str(mapping[idx])) for idx in ends)), int(bond.GetStereo())))
        else:
            return None
    return tuple(sorted(records))


def stereo_numbering(mol, mapping):
    """P-14.4(j), only after chemical and isotope numbering have tied.

    Use only specified original configurations. Unspecified units acquire no
    invented descriptor. This is numbering of one physical parent, not the
    P-44 comparison between different parent structures/CIP sequences.
    """
    Chem, stereo = _stereo_runtime()
    comparison = _comparison_module()
    locant_key, locant_set = comparison.locant_key, comparison.locant_set

    units = [(mapping[i], d) for i, d in stereo.assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True)
             if i in mapping]
    for a, b, descriptor in stereo._indexed_bond_stereochemistry(mol, Chem):
        if {a, b} & mapping.keys():
            if not {a, b} <= mapping.keys():
                return None
            units.append((min((mapping[a], mapping[b]), key=locant_key), descriptor))
    if not units:
        return ()
    if any(locant_key(loc) is None or d not in {'R', 'S', 'r', 's', 'Z', 'E'} for loc, d in units):
        return None
    ordered = sorted(units, key=lambda row: locant_key(row[0]))
    return (locant_set(loc for loc, _ in ordered),
            tuple({'Z':0,'E':1,'R':0,'r':1,'S':2,'s':3}[d] for _, d in ordered))
