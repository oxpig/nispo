"""Graph-based locant omission for homogeneous hydrides under P-14.3.4.2."""
from .number_terms import basic_number_term


# Mononuclear stems from P-21.1. Boron hydrides have separate rules.
_STEMS = {6: ('', 4), 7: ('az', 3), 14: ('sil', 4), 15: ('phosph', 3),
          13: ('alum', 3), 31: ('gall', 3), 32: ('germ', 4), 33: ('ars', 3),
          49: ('indig', 3), 50: ('stann', 4), 51: ('stib', 3),
          81: ('thall', 3), 82: ('plumb', 4), 83: ('bismuth', 3)}


def organyl_hydride_parent(mol, Chem):
    """A mononuclear Group 13 or heavier Group 14 hydride bearing mapped organyls and halogens."""
    from .component_mapping import mapped_general_branch
    from .constants import HALO, ROOTS
    from .functional_parent_extensions import side
    from .organyl_prefix_components import organyl_prefix_nature
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    from .substituent_selection import mapped_saturated_carbon_prefix

    center = None
    saturated = True
    for a in mol.GetAtoms():
        if a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery():
            return ''
        if a.GetAtomicNum() == 6:
            if a.GetTotalValence() != 4:
                return ''
        elif a.GetSymbol() in HALO and a.GetDegree() == a.GetTotalValence() == 1:
            saturated = False
        elif (center is None and not a.IsInRing()
              and (a.GetAtomicNum() in (5, 13, 31, 49, 81) and a.GetTotalValence() == 3
                   or a.GetAtomicNum() in (14, 32, 50, 82) and a.GetTotalValence() == 4)):
            center = a
        else:
            return ''
    if (center is None or not center.GetDegree() or mol.GetStereoGroups()
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in center.GetBonds())
            or len(Chem.GetMolFrags(mol)) != 1):
        return ''
    for b in mol.GetBonds():
        if b.HasQuery() or b.GetStereo() or b.GetBondTypeAsDouble() not in (1, 1.5, 2, 3):
            return ''
        if b.GetBondType() != Chem.BondType.SINGLE:
            saturated = False
    saturated = saturated and not mol.GetRingInfo().NumRings()
    host = center.GetIdx()
    units = []
    for atom in center.GetNeighbors():
        root = atom.GetIdx()
        if atom.GetSymbol() in HALO:
            units.append(PrefixUnit(HALO[atom.GetSymbol()], 'simple'))
            continue
        if atom.GetDegree() == 1:
            units.append(PrefixUnit('methyl', 'simple'))
            continue
        scope = side(mol, root, host)
        if saturated and len(scope) in ROOTS and all(mol.GetAtomWithIdx(i).GetDegree() <= 2 for i in scope):
            # The bond to the center contributes to the root's degree. This
            # test therefore proves an unbranched chain attached at its end.
            units.append(PrefixUnit(ROOTS[len(scope)] + 'yl', 'simple'))
            continue
        branch = (mapped_saturated_carbon_prefix(mol, Chem, root, host, scope) if saturated
                  else mapped_general_branch(mol, Chem, root, host))
        if not branch or not branch[0] or branch[1] != scope:
            return ''
        text = branch[0]
        nature = organyl_prefix_nature(mol, root, scope, text)
        if not nature:
            return ''
        units.append(PrefixUnit(text, nature))
    sequence = mononuclear_prefix_sequence(units)
    stem = 'bor' if center.GetAtomicNum() == 5 else _STEMS[center.GetAtomicNum()][0]
    return sequence.text + stem + 'ane' if sequence else ''


def halogenated_short_parent(mol):
    """Mono- and dinuclear hydrides with terminal halogens as their only ligands."""
    from .constants import HALO
    from .formatting import format_prefixes

    if not 2 <= mol.GetNumAtoms() <= 8 or mol.GetStereoGroups():
        return ''
    parents, halogens, labelled_hydrogens = [], [], []
    annotated = False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.HasQuery() or atom.IsInRing():
            return ''
        if atom.GetIsotope() or atom.GetChiralTag():
            annotated = True
        if atom.GetSymbol() in HALO:
            halogens.append(atom)
        elif atom.GetAtomicNum() == 1 and atom.GetIsotope():
            labelled_hydrogens.append(atom)
        else:
            parents.append(atom)
    if (len(parents) not in (1, 2) or not halogens or
            len({a.GetAtomicNum() for a in parents}) != 1):
        return ''
    element = parents[0].GetAtomicNum()
    pnictogen_pair = element in (7, 15, 33, 51, 83) and len(parents) == 2 and len(halogens) == 1
    if element not in (6, 14, 32, 50, 82) and not pnictogen_pair:
        return ''
    if any(a.GetTotalValence() != _STEMS[element][1] for a in parents):
        return ''
    sites = {a.GetIdx(): j for j, a in enumerate(parents, 1)}
    if any(h.GetDegree() != 1 or h.GetTotalValence() != 1
           or h.GetNeighbors()[0].GetIdx() not in sites for h in labelled_hydrogens):
        return ''
    ligands = []
    for atom in halogens:
        if atom.GetDegree() != 1 or atom.GetTotalValence() != 1:
            return ''
        host = atom.GetNeighbors()[0].GetIdx()
        if host not in sites:
            return ''
        ligands.append((host, atom))
    for b in mol.GetBonds():
        if b.HasQuery():
            return ''
        if b.GetStereo():
            annotated = True
    stem = _STEMS[element][0]
    if len(sites) == 1:
        parent = ('meth' if element == 6 else stem) + 'ane'
    else:
        bond = mol.GetBondBetweenAtoms(*(a.GetIdx() for a in parents))
        if bond is None or bond.GetBondTypeAsDouble() not in (1, 2, 3):
            return ''
        degree = int(bond.GetBondTypeAsDouble())
        if pnictogen_pair and degree != 1:
            return ''
        parent = ('eth' if element == 6 else 'di' + stem) + ('ane', 'ene', 'yne')[degree - 1]
        if element == 7:
            parent = 'hydrazine'
    if annotated:
        return _annotated_short_parent(mol, parent, parents, ligands, labelled_hydrogens)
    rows = [(sites[host], HALO[atom.GetSymbol()]) for host, atom in ligands]
    if len(sites) == 2:
        rows = min((rows, [(3 - i, name) for i, name in rows]), key=_prefix_numbering)
    omit = len(sites) == 1 or len(rows) == 1
    return format_prefixes([(str(i), name) for i, name in rows], omit_locants=omit,
                           separate_unlocanted=len(sites) == 1) + parent


def _prefix_numbering(rows):
    return sorted(i for i, _ in rows), sorted((name, i) for i, name in rows)


def _annotated_short_parent(mol, parent, parents, ligands, hydrogens):
    """Apply P-82 and whole-graph CIP after numbering the halogen substituents."""
    from rdkit import Chem
    from .component_mapping import MappedComponent, _mapped_isotope_descriptor_plan, hydrogen_bindings
    from .constants import HALO
    from .formatting import format_prefixes
    from .parent_annotations import isotope_numbering_key
    from .stereochemistry import assigned_atom_stereochemistry, assigned_bond_stereochemistry

    ids = tuple(a.GetIdx() for a in parents)
    tagged = {a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag()}
    assigned = dict(assigned_atom_stereochemistry(mol, Chem)) if tagged else {}
    if (tagged != assigned.keys() or not tagged <= set(ids)
            or any(d not in ('R', 'S') for d in assigned.values())):
        return ''
    configured = {b.GetIdx() for b in mol.GetBonds() if b.GetStereo()}
    bonds = assigned_bond_stereochemistry(mol, Chem) if configured else ()
    if ({mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b, _ in bonds} != configured
            or any({a, b} != set(ids) or d not in ('E', 'Z') for a, b, d in bonds)):
        return ''
    # P-82.2.1 requires a count of one only where several H can be replaced
    # at that site. Count implicit and explicitly drawn H on the original host.
    unique_h = {h.GetIdx() for h in hydrogens if h.GetNeighbors()[0].GetTotalNumHs(includeNeighbors=True) == 1}
    choices = []
    for seq in (ids, ids[::-1]) if len(ids) == 2 else (ids,):
        mapping = {idx: str(j) for j, idx in enumerate(seq, 1)}
        rows, prefixes = [], []
        for host, atom in ligands:
            symbol = atom.GetSymbol()
            rows.append((mapping[host], HALO[symbol]))
            label = f'({atom.GetIsotope()}{symbol})' if atom.GetIsotope() else ''
            prefixes.append((mapping[host], label + HALO[symbol]))
        isotope_mapping = hydrogen_bindings(mol, mapping) if hydrogens else mapping
        isotope_key = isotope_numbering_key(mol, isotope_mapping, include_substituent_groups=True)
        stereo = sorted((mapping[i], d) for i, d in assigned.items())
        stereo.extend(('1', d) for _a, _b, d in bonds)
        component = MappedComponent(parent, tuple(isotope_mapping.items()), frozenset(isotope_mapping),
                                    prefixes=tuple(prefixes))
        isotopes, claims = _mapped_isotope_descriptor_plan(mol, Chem, component, isotope_mapping,
                                                          sole_hydrogen_atoms=unique_h)
        omit = len(ids) == 1 or len(ligands) == 1 and not any(cited for *_, cited in claims)
        isotope_text = '(' + ','.join(isotopes) + ')' if isotopes else ''
        stereo_text = ','.join(('' if len(ids) == 1 else loc) + d for loc, d in stereo)
        prefix = format_prefixes(prefixes, omit_locants=omit, hierarchical_enclosures=True,
                                 separate_unlocanted=len(ids) == 1)
        name = (f'({stereo_text})-' if stereo_text else '') + prefix + isotope_text + parent
        choices.append(((_prefix_numbering(rows), isotope_key, stereo), name))
    best = min(key for key, _ in choices)
    winners = {name for key, name in choices if key == best}
    return winners.pop() if len(winners) == 1 else ''


def simple_carbocycle(mol, Chem, ring):
    """P-14.3.4.2(c,d) for one cyclic feature with no competing locants."""
    from .constants import HALO, ROOTS
    from .formatting import attach_suffix

    size = len(ring)
    if size not in ROOTS or mol.GetNumAtoms() not in (size, size + 1) or mol.GetStereoGroups():
        return ''
    core = frozenset(ring)
    outside = None
    for atom in mol.GetAtoms():
        if (atom.GetIsotope() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetChiralTag() or atom.HasQuery() or atom.GetIsAromatic()):
            return ''
        if atom.GetIdx() in core:
            if atom.GetAtomicNum() != 6 or atom.GetTotalValence() != 4:
                return ''
        else:
            outside = atom
    if mol.GetNumBonds() != mol.GetNumAtoms():
        return ''
    multiple = []
    for bond in mol.GetBonds():
        if bond.HasQuery() or bond.GetBondTypeAsDouble() not in (1, 2, 3):
            return ''
        if bond.GetBondTypeAsDouble() != 1:
            multiple.append(bond)
    stem = 'cyclo' + ROOTS[size]
    if outside is not None:
        if multiple or outside.GetDegree() != 1:
            return ''
        symbol = outside.GetSymbol()
        if symbol in HALO and not outside.GetTotalNumHs():
            return HALO[symbol] + stem + 'ane'
        if symbol in ('O', 'S') and outside.GetTotalNumHs() == 1:
            return attach_suffix(stem + 'ane', 'ol' if symbol == 'O' else 'thiol')
        return ''
    if len(multiple) != 1:
        return ''
    bond = multiple[0]
    stereo = ''
    if bond.GetStereo():
        from .stereochemistry import assigned_bond_stereochemistry

        descriptors = assigned_bond_stereochemistry(mol, Chem)
        if len(descriptors) != 1 or descriptors[0][2] not in ('E', 'Z'):
            return ''
        stereo = '(' + descriptors[0][2] + ')-'
    return stereo + stem + ('ene' if bond.GetBondTypeAsDouble() == 2 else 'yne')


def monounsaturated_chain(mol):
    """Return the complete parent name only when the graph is the bare chain."""
    atoms = tuple(mol.GetAtoms())
    if len(atoms) < 2 or mol.GetNumBonds() != len(atoms) - 1 or mol.GetStereoGroups():
        return ''
    element = atoms[0].GetAtomicNum()
    if element not in _STEMS or (element == 6 and len(atoms) > 3):
        return ''
    stem, valence = _STEMS[element]
    if any(a.GetAtomicNum() != element or a.GetDegree() not in (1, 2)
           or a.GetTotalValence() != valence or a.GetIsotope() or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery()
           or a.IsInRing() for a in atoms):
        return ''
    positions = []
    end = next(a for a in atoms if a.GetDegree() == 1)
    previous, current = -1, end
    for locant in range(1, len(atoms)):
        bond = next(b for b in current.GetBonds() if b.GetOtherAtomIdx(current.GetIdx()) != previous)
        order = bond.GetBondTypeAsDouble()
        if order not in (1, 2, 3) or bond.HasQuery() or (order != 2 and bond.GetStereo()):
            return ''
        if order != 1:
            positions.append((min(locant, len(atoms) - locant), order, bond))
        previous, current = current.GetIdx(), bond.GetOtherAtom(current)
    if len(positions) != 1:
        return ''
    position, order, bond = positions[0]
    if not any(a.GetTotalNumHs() for a in atoms):
        return ''  # Elemental molecules such as dinitrogen have their own names.
    if element == 6:
        if len(atoms) == 2 and order == 3:
            return 'acetylene'
        base = 'eth' if len(atoms) == 2 else 'prop'
    else:
        multiplier = basic_number_term(len(atoms))
        if not multiplier:
            return ''
        base = multiplier + stem
    locant_text = '' if len(atoms) <= 3 else f'-{position}-'
    stereo = ''
    if bond.GetStereo():
        from rdkit import Chem
        from .stereochemistry import assigned_bond_stereochemistry

        descriptors = assigned_bond_stereochemistry(mol, Chem)
        if len(descriptors) != 1:
            return ''
        stereo = '(' + (str(position) if locant_text else '') + descriptors[0][2] + ')-'
    return stereo + base + locant_text + ('ene' if order == 2 else 'yne')
