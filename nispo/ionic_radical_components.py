"""Localized ionic hydrides and carbonyl dipoles with mapped ligands."""

from dataclasses import replace

from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
from .formatting import enclose_name_unit, format_prefixes
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_PARENTS = {(6, -1): (3, 'methanide'), (6, 1): (3, 'methylium'),
            (7, -1): (2, 'azanide'), (7, 1): (4, 'azanium'),
            (8, 1): (3, 'oxidanium'), (16, 1): (3, 'sulfanium'),
            (34, 1): (3, 'selanium'), (52, 1): (3, 'tellanium'),
            (5, -1): (4, 'boranuide'), (15, 1): (4, 'phosphanium'),
            (33, 1): (4, 'arsanium'), (51, 1): (4, 'stibanium'),
            (8, -1): (1, 'oxidanide'), (16, -1): (1, 'sulfanide'),
            (34, -1): (1, 'selanide'), (52, -1): (1, 'tellanide'),
            (9, 1): (2, 'fluoranium'), (17, 1): (2, 'chloranium'),
            (35, 1): (2, 'bromanium'), (53, 1): (2, 'iodanium'),
            (85, 1): (2, 'astatanium'),
            (9, -1): (0, 'fluoride'), (17, -1): (0, 'chloride'),
            (35, -1): (0, 'bromide'), (53, -1): (0, 'iodide'),
            (85, -1): (0, 'astatide')}


def localized_ionic_radical(mol, Chem, *, allow_closed_shell=False):
    if mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    radicals = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if allow_closed_shell and len(charged) == 2 and not radicals:
        return _onium_ylide(mol, Chem, charged)
    if (len(charged) != 1 or len(radicals) > 1
            or radicals and charged[0].GetIdx() != radicals[0].GetIdx()
            or not radicals and not allow_closed_shell):
        return None
    center = charged[0]
    spec = _PARENTS.get((center.GetAtomicNum(), center.GetFormalCharge()))
    return _localized_ionic_hydride(mol, Chem, center, spec, allow_closed_shell)


def substituted_sulfanuide(mol, Chem):
    """Use hydride addition to divalent sulfane for a two-arm S(-1)H."""
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetChiralTag()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16
               and a.GetFormalCharge() == -1 and a.GetTotalValence() == 3
               and a.GetDegree() == 2 and a.GetTotalNumHs() == 1
               and all(n.GetAtomicNum() == 6 for n in a.GetNeighbors())
               and all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds())]
    if len(centers) != 1:
        return None
    center = centers[0]
    others = [a for a in mol.GetAtoms() if a.GetFormalCharge() and a.GetIdx() != center.GetIdx()]
    if len(others) > 1 or any(
            a.GetAtomicNum() != 16 or a.GetFormalCharge() != -1 or a.GetDegree() != 1
            or a.GetTotalNumHs() or a.GetTotalValence() != 1
            or a.GetNeighbors()[0].GetAtomicNum() != 6 for a in others):
        return None
    decision = _localized_ionic_hydride(mol, Chem, center, (3, 'sulfanuide'), True)
    if decision is None:
        return None
    rule = 'P-72.3/P-72.8.1'
    if others:
        issue = SelectionLimitation('dianion-parent', 'P-72.5/P-72.7',
            'The localized sulfanuide and terminal anion are represented. '
            'Preferred comparison of dianion parents and suffixes remains open.')
    else:
        issue = SelectionLimitation('sulfanuide-substituents', 'P-29/P-46',
            'Hydride addition establishes the local sulfanuide parent form. '
            'Preferred comparison of both carbon substituent prefixes remains open.')
    return replace(decision, rule=rule + '/P-29', preference_issues=(issue,))


def _onium_ylide(mol, Chem, charged):
    """P-74: build a carbanion bearing an explicitly charged onium substituent."""
    carbon, onium = sorted(charged, key=lambda atom: atom.GetFormalCharge())
    if (carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() != -1
            or onium.GetFormalCharge() != 1 or onium.GetAtomicNum() not in {7, 8, 15, 16, 33, 34, 51, 52}
            or onium.IsInRing() or onium.GetChiralTag() or onium.GetIsotope()):
        return None
    root, ion = carbon.GetIdx(), onium.GetIdx()
    bond = mol.GetBondBetweenAtoms(root, ion)
    valence, parent = _PARENTS[onium.GetAtomicNum(), 1]
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or onium.GetTotalValence() != valence:
        return None
    from .component_mapping import mapped_general_branch

    ligands, owned = [], {ion}
    for edge in onium.GetBonds():
        other = edge.GetOtherAtom(onium)
        if other.GetIdx() == root:
            continue
        if other.GetAtomicNum() == 1 or edge.GetStereo():
            return None
        name, atoms = mapped_branch(mol, Chem, other.GetIdx(), ion)
        if not name:
            name, atoms = mapped_general_branch(mol, Chem, other.GetIdx(), ion)
        if not name or root in atoms or owned.intersection(atoms):
            return None
        order = edge.GetBondTypeAsDouble()
        if (order not in {1, 2, 3} or order == 2 and not name.endswith("ylidene")
                or order == 3 and not name.endswith("ylidyne")):
            return None
        ligands.append(('', name))
        owned.update(atoms)
    head = format_prefixes(ligands, omit_locants=True, hierarchical_enclosures=True, separate_unlocanted=True)
    ligand = ion, head + parent + 'yl', frozenset(owned)
    return _localized_ionic_hydride(mol, Chem, carbon, _PARENTS[6, -1], True, ionic_ligand=ligand)


def _localized_ionic_hydride(mol, Chem, center, spec, allow_closed_shell, *, ionic_ligand=None):
    count = center.GetNumRadicalElectrons()
    if (spec is None or count not in ({0, 1, 2} if allow_closed_shell else {1, 2})
            or center.IsInRing() or int(center.GetChiralTag())
            or center.GetTotalValence() != spec[0] - count
            or any(b.GetBondTypeAsDouble() not in {1, 2, 3} for b in center.GetBonds())):
        return None
    root = center.GetIdx()
    bindings = hydrogen_bindings(mol, {root: ''})
    prefixes, embedded, roles = [], set(), []
    for neighbor in center.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            continue
        if ionic_ligand is not None and neighbor.GetIdx() == ionic_ligand[0]:
            _, text, owned = ionic_ligand
        elif (neighbor.GetAtomicNum() in {8, 16} and neighbor.GetFormalCharge() == -1
                and neighbor.GetDegree() == 1 and not neighbor.GetTotalNumHs()
                and not neighbor.GetIsotope() and not neighbor.GetNumRadicalElectrons()
                and mol.GetBondBetweenAtoms(root, neighbor.GetIdx()).GetBondTypeAsDouble() == 1):
            text = 'oxido' if neighbor.GetAtomicNum() == 8 else 'sulfido'
            owned = frozenset({neighbor.GetIdx()})
        else:
            text, owned = mapped_branch(mol, Chem, neighbor.GetIdx(), root)
        if not text or owned & (bindings.keys() | embedded):
            return None
        order = mol.GetBondBetweenAtoms(root, neighbor.GetIdx()).GetBondTypeAsDouble()
        if order > 1 and not (
                text.endswith('ylidene' if order == 2 else 'ylidyne')
                or order == 2 and (text == 'oxo' or text.endswith('imino'))):
            return None
        prefixes.append(('', text))
        embedded.update(owned)
        roles.append(('charged-substituent' if any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in owned)
                      else 'neutral-substituent', owned, ()))
    required = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 or a.GetIsotope()}
    if bindings.keys() | embedded != required:
        return None
    parent = spec[1] if count == 0 else spec[1].removesuffix('e') + ('yl' if count == 1 else 'ylidene')
    name = decorate_component(mol, Chem, MappedComponent(parent, tuple(bindings.items()),
        frozenset(bindings), tuple(prefixes), frozenset(embedded),
        prefix_mode='mononuclear', systematic_parent=True))
    if not name:
        return None
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if count:
        rule = 'P-75.2/P-72/P-73/P-29/P-82'
        role = 'localized-ionic-radical'
        issue = SelectionLimitation('ionic-radical-parent', 'P-75.2/P-75.3/P-75.4',
            'The localized radical ion and its substituents are represented. '
            'Preferred comparison with functional radical-ion names remains open.')
    elif ionic_ligand is not None:
        rule = 'P-74.1.3/P-74.2/P-72.2/P-73.6'
        role = 'carbanionic-parent'
        issue = SelectionLimitation('ylide-parent', rule,
            'Both ionic centres are mapped. Comparison with other dipolar naming methods remains open.')
    else:
        rule = 'P-72/P-73/P-29/P-82'
        role = 'localized-ionic-center'
        issue = SelectionLimitation('localized-ionic-parent', 'P-72/P-73',
            'The localized ion and its substituents are represented. '
            'Preferred comparison with other ionic parents remains open.')
    return MethodDecision(name=name, method='substitutive', principal_centers=frozenset({root}),
        atoms=heavy, rule=rule, atom_locants=tuple(bindings.items()),
        component_roles=((role, frozenset({root}), ((root, ''),)), *roles),
        preference_issues=(issue,))


def carbonyl_oxidanium(mol, Chem, *, dipole_parent=False):
    """Name a mapped carbonyl oxide or a localized O–S dipole."""
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetChiralTag()
                   or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8
               and a.GetFormalCharge() == 1 and a.GetDegree() == 2 and a.GetTotalValence() == 3]
    if len(centers) != 1:
        return None
    center = centers[0]
    carbons = [b.GetOtherAtom(center) for b in center.GetBonds()
               if b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(center).GetAtomicNum() == 6]
    anions = [b.GetOtherAtom(center) for b in center.GetBonds()
              if b.GetBondType() == Chem.BondType.SINGLE
              and b.GetOtherAtom(center).GetAtomicNum() in {8, 16}
              and b.GetOtherAtom(center).GetFormalCharge() == -1
              and b.GetOtherAtom(center).GetDegree() == 1
              and not b.GetOtherAtom(center).GetTotalNumHs()]
    if len(carbons) != 1 or len(anions) != 1:
        return None
    terminal = anions[0]
    allowed = {center.GetIdx(), terminal.GetIdx()}
    charged = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge()}
    if dipole_parent:
        if charged != allowed:
            return None
        return _carbonyl_dipole_parent(mol, Chem, carbons[0], center, terminal)
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() and atom.GetIdx() not in allowed:
            if (atom.GetAtomicNum() != 8 or atom.GetFormalCharge() != -1
                    or atom.GetDegree() != 1 or atom.GetTotalNumHs()
                    or atom.GetNeighbors()[0].GetIdx() != carbons[0].GetIdx()
                    or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE):
                return None
    decision = _localized_ionic_hydride(mol, Chem, center, (3, 'oxidanium'), True)
    if decision is None:
        return None
    return replace(decision, rule='P-73.1.1.2/P-72.6.2/P-29',
        preference_issues=(SelectionLimitation('mixed-charge-carbonyl-parent', 'P-73.6/P-74.0/P-74.1.3/P-72.7',
            'The localized oxidanium form represents all charged sites. '
            'Preferred selection of a parent that includes both charge types, or of an anionic parent, remains open.'),))


def _carbonyl_dipole_parent(mol, Chem, carbon, center, terminal):
    center_idx, terminal_idx = center.GetIdx(), terminal.GetIdx()
    branch, owned = mapped_branch(mol, Chem, carbon.GetIdx(), center_idx)
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if not branch.endswith('ylidene') or owned != heavy - {center_idx, terminal_idx}:
        return None

    if terminal.GetAtomicNum() == 8:
        rule = 'P-74.2.2.1.6/P-29'
        locants = ((center_idx, '2'), (terminal_idx, '1'))
        role = ('dioxidanium-ide', frozenset({center_idx, terminal_idx}), locants)
        ligand = owned
        name = format_prefixes([('2', branch)], hierarchical_enclosures=True) + 'dioxidan-2-ium-1-ide'
        issue = SelectionLimitation('carbonyl-ylidene-prefix', 'P-29/P-46',
            'The carbonyl oxide uses the intact dioxidanium-ide parent. '
            'Preferred comparison of the mapped carbon substituent prefix remains open.')
    else:
        rule = 'P-73.6/P-74.1.3/P-29'
        locants = ((terminal_idx, ''),)
        role = ('sulfanide', frozenset({terminal_idx}), locants)
        ligand = owned | {center_idx}
        prefix = format_prefixes([('', branch)], omit_locants=True, hierarchical_enclosures=True)
        name = enclose_name_unit(prefix + 'oxidaniumyl') + 'sulfanide'
        issue = SelectionLimitation('carbonyl-O-S-dipole-parent', 'P-74.1.3/P-74.2.2',
            'The O–S dipole is expressed on an anionic parent. '
            'Preferred comparison with intact dipolar parents remains open.')
    return MethodDecision(name=name, method='substitutive',
        principal_centers=frozenset({center_idx, terminal_idx}), atoms=heavy, rule=rule,
        atom_locants=locants, component_roles=(role, ('carbonyl-ligand', ligand, ())),
        preference_issues=(issue,))


def mixed_cycloalkyl_oxidanium(mol, Chem):
    """Construct a source-bounded mixed cycloalkyl O+ parent.

    Each arm is a plain saturated carbon chain, a saturated monocycle of
    three through six carbons, or phenyl. At least one arm is a saturated
    monocycle and at least two arms are inequivalent. This is a structural
    proof of separately attached P-16.5.1.3.1 components, not a preferred
    whole-name parent/method comparison. Annotated graphs keep their existing
    routes rather than being projected onto this scope.
    """
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups():
        return None
    atoms = tuple(mol.GetAtoms())
    charged = [atom for atom in atoms if atom.GetFormalCharge()]
    if len(charged) != 1:
        return None
    center = charged[0]
    if (center.GetAtomicNum() != 8 or center.GetFormalCharge() != 1
            or center.IsInRing() or center.GetNumRadicalElectrons()
            or center.GetTotalValence() != 3 or center.GetDegree() not in (2, 3)
            or any(atom.GetAtomicNum() != 6 for atom in center.GetNeighbors())
            or any(atom.GetAtomicNum() not in (6, 8) or atom.GetIsotope()
                   or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                   for atom in atoms)
            or any(atom.GetAtomicNum() == 8 and atom.GetIdx() != center.GetIdx()
                   for atom in atoms)
            or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                   for bond in mol.GetBonds())):
        return None
    root = center.GetIdx()
    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in center.GetBonds()):
        return None

    roots = tuple(atom.GetIdx() for atom in center.GetNeighbors())
    carbon_arms = []
    has_saturated_cycle = False
    for attachment in roots:
        seen, pending = {attachment}, [attachment]
        while pending:
            index = pending.pop()
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
                other = neighbor.GetIdx()
                if other != root and other not in seen:
                    seen.add(other)
                    pending.append(other)
        arm = frozenset(seen)
        if any(arm & earlier for earlier in carbon_arms):
            return None
        external = {(i, n.GetIdx()) for i in arm
                    for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in arm}
        if external != {(attachment, root)}:
            return None
        carbon_arms.append(arm)
        internal = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                             if n.GetIdx() in arm) for i in arm}
        if any(mol.GetAtomWithIdx(i).GetTotalValence() != 4 for i in arm):
            return None
        ring_atoms = {i for i in arm if mol.GetAtomWithIdx(i).IsInRing()}
        if ring_atoms:
            if ring_atoms != set(arm) or any(len(internal[i]) != 2 for i in arm):
                return None
            aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in arm)
            if aromatic:
                if len(arm) != 6 or any(not mol.GetBondBetweenAtoms(i, j).GetIsAromatic()
                                        for i in arm for j in internal[i]):
                    return None
            else:
                if (len(arm) not in (3, 4, 5, 6)
                        or any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in arm)
                        or any(mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE
                               for i in arm for j in internal[i])):
                    return None
                has_saturated_cycle = True
        elif (len(internal[attachment]) > 1
              or any(len(neighbors) > 2 for neighbors in internal.values())
              or any(mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE
                     for i in arm for j in internal[i])):
            return None
    if not has_saturated_cycle or frozenset().union(*carbon_arms) | {root} != frozenset(range(len(atoms))):
        return None
    # Canonical symmetry classes compare rooted arm identity without making
    # input atom numbers or optional caller atom maps a chemical tie-break.
    unlabelled = Chem.Mol(mol)
    for atom in unlabelled.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = tuple(Chem.CanonicalRankAtoms(unlabelled, breakTies=False))
    if len({ranks[index] for index in roots}) < 2:
        return None

    decision = localized_ionic_radical(mol, Chem, allow_closed_shell=True)
    if decision is None or decision.principal_centers != frozenset({root}):
        return None
    owned = [frozenset(role[1]) for role in decision.component_roles]
    if (decision.atoms != frozenset(range(len(atoms))) or not owned
            or frozenset().union(*owned) != decision.atoms
            or sum(map(len, owned)) != len(decision.atoms)):
        return None
    return decision


def one_to_one_ionic_salt(mol, Chem, name_atomic_counterion):
    """P-77 one +1 component and one -1 component, each explicitly owned."""
    parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    original_parts = Chem.GetMolFrags(mol)
    if len(parts) != 2 or sorted(Chem.GetFormalCharge(p) for p in parts) != [-1, 1]:
        return None
    words, roles = {}, []
    for part, original_ids in zip(parts, original_parts):
        charge = Chem.GetFormalCharge(part)
        decision = localized_ionic_radical(part, Chem, allow_closed_shell=True)
        name = decision.name if decision is not None else ''
        if not name and part.GetNumAtoms() == 1:
            atom = part.GetAtomWithIdx(0)
            if (not atom.GetIsotope() and not atom.GetNumRadicalElectrons()
                    and not atom.GetTotalNumHs()):
                name = name_atomic_counterion(part)
        if not name:
            return None
        words[charge] = name
        roles.append(('ionic-component', frozenset(i for i in original_ids
                      if mol.GetAtomWithIdx(i).GetAtomicNum() > 1), ()))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if frozenset().union(*(r[1] for r in roles)) != heavy:
        return None
    return MethodDecision(name=words[1] + ' ' + words[-1], method='functional-class',
        principal_centers=frozenset(), atoms=heavy, rule='P-77/P-72/P-73',
        component_roles=tuple(roles))
