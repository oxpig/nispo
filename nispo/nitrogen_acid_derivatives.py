"""Original-graph N substitution for carbon acid derivatives (P-66)."""

from .branch_rules import collect_subtree
from .component_mapping import MappedComponent, decorate_component, mapped_general_branch
from .formatting import (attach_suffix, enclose_name_unit, format_prefixes,
                         join_prefix_parent, prefix_alphabetization_key)
from .functional_inventory import CharacteristicGroup
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _ligands(mol, Chem, nitrogen, center, owned, roles, *, other_core=(),
             branch_resolver=mapped_general_branch):
    texts = []
    for atom in mol.GetAtomWithIdx(nitrogen).GetNeighbors():
        root = atom.GetIdx()
        if root == center or root in other_core or atom.GetAtomicNum() == 1:
            continue
        if mol.GetBondBetweenAtoms(nitrogen, root).GetBondTypeAsDouble() != 1:
            return None
        text, atoms = branch_resolver(mol, Chem, root, nitrogen)
        if not text or atoms & owned:
            return None
        texts.append(text)
        owned.update(atoms)
        roles.append(('N-substituent', atoms, ()))
    return texts


def _local_configuration(mol, Chem, center, imino):
    bond = mol.GetBondBetweenAtoms(center, imino)
    if not bond.GetStereo():
        return ''
    from .stereochemistry import assigned_bond_stereochemistry

    return next((d for a, b, d in assigned_bond_stereochemistry(mol, Chem)
                 if {a, b} == {center, imino}), None)


def _record_configuration(mol, center, imino, text, configuration, atoms):
    if configuration:
        from .molecule_cache import mol_cache_dict

        mol_cache_dict(mol, 'rendered_mapped_branch_bond_stereo')[
            (center, imino, text, frozenset(atoms))] = configuration


def amidino_prefix(mol, Chem, root, blocked, atoms):
    """An N-rooted amidine/guanidine arm keeps its drawn C=N bond and ligands."""
    nitrogen = mol.GetAtomWithIdx(root)
    if (nitrogen.IsInRing() or nitrogen.GetFormalCharge() or nitrogen.GetIsotope()
            or mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() != 1):
        return None
    for carbon in nitrogen.GetNeighbors():
        center = carbon.GetIdx()
        if (center == blocked or carbon.GetAtomicNum() != 6 or carbon.IsInRing()
                or mol.GetBondBetweenAtoms(root, center).GetBondTypeAsDouble() != 1):
            continue
        imino = [bond.GetOtherAtomIdx(center) for bond in carbon.GetBonds()
                 if bond.GetBondTypeAsDouble() == 2 and bond.GetOtherAtom(carbon).GetAtomicNum() == 7]
        if len(imino) != 1 or mol.GetAtomWithIdx(imino[0]).GetIsotope():
            continue
        imino = imino[0]
        if (nitrogen.GetTotalValence() != 3 or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
                or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                       or a.GetIsotope() for a in (carbon, mol.GetAtomWithIdx(imino)))):
            continue
        owned = {root, center, imino}
        imino_texts = _ligands(mol, Chem, imino, center, owned, [])
        if imino_texts is None or len(imino_texts) > 1:
            return None
        imino_text = 'imino'
        if imino_texts:
            ligand = imino_texts[0]
            imino_text = (ligand if ligand == 'cyano' else enclose_name_unit(ligand)) + 'imino'
        carbon_prefixes = [("1", imino_text)]
        for child in carbon.GetNeighbors():
            idx = child.GetIdx()
            if idx in owned or child.GetAtomicNum() == 1:
                continue
            text, branch = mapped_general_branch(mol, Chem, idx, center)
            if not text or branch & owned or not branch <= atoms:
                return None
            owned.update(branch)
            carbon_prefixes.append(('1', text))
        if owned != atoms:
            return None
        configuration = _local_configuration(mol, Chem, center, imino)
        if configuration is None:
            return None
        if not configuration and len(carbon_prefixes) == 2 and (
                '1', 'amino') in carbon_prefixes and not imino_texts:
            return 'carbamimidoylamino', atoms
        ylmethyl = format_prefixes(carbon_prefixes, omit_locants=True,
                                  hierarchical_enclosures=True,
                                  separate_unlocanted=True) + 'methyl'
        # A single-carbon component takes an unlocanted E/Z descriptor.
        if configuration:
            ylmethyl = f'({configuration})-' + ylmethyl
        text = enclose_name_unit(ylmethyl) + 'amino'
        _record_configuration(mol, center, imino, text, configuration, owned)
        return text, atoms
    return None


def _amide_nitrogen_roles(mol, group):
    core = frozenset(group.atoms)
    if group.kind == 'cyanamide':
        return core, None, (group.ligand,)
    if group.kind == 'cyanohydrazide':
        terminal = next(i for i in core - {group.center, group.ligand}
                        if mol.GetBondBetweenAtoms(i, group.ligand) is not None)
        return core, None, (group.ligand, terminal)
    imino = next(i for i in core - {group.center}
                 if mol.GetBondBetweenAtoms(group.center, i).GetBondTypeAsDouble() == 2)
    return core, imino, tuple(sorted(core - {group.center, imino}))


def sulfonimidamide_site(sulfur):
    """Recognize the neutral R–S(=O)(=N)–N core on original atoms."""
    if (sulfur.GetAtomicNum() != 16 or sulfur.IsInRing() or sulfur.GetFormalCharge()
            or sulfur.GetDegree() != 4
            or sulfur.GetNumRadicalElectrons()):
        return None
    center = sulfur.GetIdx()
    roles = {(bond.GetOtherAtom(sulfur).GetAtomicNum(), bond.GetBondTypeAsDouble()):
             bond.GetOtherAtom(sulfur) for bond in sulfur.GetBonds()}
    expected = {(6, 1), (8, 2), (7, 1), (7, 2)}
    if (roles.keys() != expected or roles[8, 2].GetDegree() != 1
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in roles.values())
            or any(roles[key].GetTotalValence() != 3 for key in ((7, 1), (7, 2)))):
        return None
    return CharacteristicGroup('sulfonimidamide', center,
        (center, roles[8, 2].GetIdx(), roles[7, 1].GetIdx(), roles[7, 2].GetIdx()),
        roles[6, 1].GetIdx(), roles[7, 1].GetIdx())


def sulfonimidamide_decision(mol, Chem, inventory, *, branch_resolver=mapped_general_branch):
    """Join the sulfur acid skeleton with its separately mapped N ligands."""
    if (not inventory.supported or len(inventory.principal) != 1 or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return None
    group = inventory.principal[0]
    if group.kind != 'sulfonimidamide':
        return None
    from .parent_rules import _sulfonimidamide_base_from_prefix
    from .stereochemistry import _leading_stereochemical_descriptors, assigned_atom_stereochemistry

    core = frozenset(group.atoms)
    imino = next(i for i in core if i not in {group.center, group.ligand}
                 and mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in (group.ligand, imino)):
        return None
    owned, roles, prefixes = set(core), [], []
    for nitrogen, locant in ((group.ligand, 'N'), (imino, "N'")):
        texts = _ligands(mol, Chem, nitrogen, group.center, owned, roles,
                         other_core=core, branch_resolver=branch_resolver)
        if texts is None:
            return None
        prefixes.extend((locant, text) for text in texts)
    acid_side = frozenset(collect_subtree(mol, group.owner, group.center))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if acid_side & owned or acid_side | owned != heavy:
        return None
    acid_prefix, acid_atoms = branch_resolver(mol, Chem, group.owner, group.center)
    if not acid_prefix or acid_atoms != acid_side:
        return None
    base = _sulfonimidamide_base_from_prefix(acid_prefix)
    if not base:
        return None
    descriptors, base = _leading_stereochemical_descriptors(base)
    if any(not item[0].isdigit() for item in descriptors):
        return None
    name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True), base)
    sulfur = mol.GetAtomWithIdx(group.center)
    if sulfur.GetChiralTag():
        descriptor = dict(assigned_atom_stereochemistry(mol, Chem)).get(group.center)
        if descriptor not in {'R', 'S'}:
            return None
        descriptors.insert(0, descriptor)
    if descriptors:
        name = f'({",".join(descriptors)})-' + name
    if any(int(b.GetStereo()) for b in sulfur.GetBonds()):
        return None
    roles += [('acid-organyl', acid_side, ())]
    return _decision(group, name, heavy, (), core, roles)


def nitrogen_acid_derivative_decision(mol, Chem, inventory):
    """Name the selected acyclic amidine, guanidine or cyanamide parent."""
    if not inventory.supported or len(inventory.principal) != 1 or mol.GetStereoGroups():
        return None
    group = inventory.principal[0]
    if (group.kind not in {'cyanamide', 'cyanohydrazide', 'guanidine', 'amidine'}
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return None
    core, imino, amino = _amide_nitrogen_roles(mol, group)
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in core):
        return None
    owned, roles, ligands = set(core), [], {}
    for nitrogen in (*amino, *((imino,) if imino is not None else ())):
        texts = _ligands(mol, Chem, nitrogen, group.center, owned, roles, other_core=core)
        if texts is None:
            return None
        ligands[nitrogen] = texts
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if group.kind == 'amidine':
        return _amidine_parent(mol, Chem, group, core, imino, amino[0], ligands, owned, roles, heavy)
    if owned != heavy:
        return None
    if group.kind == 'cyanamide':
        prefixes = [("N", text) for text in ligands[amino[0]]]
        name = format_prefixes(prefixes, omit_locants=True, hierarchical_enclosures=True,
                               separate_unlocanted=True) + 'cyanamide'
        configuration = ''
    elif group.kind == 'cyanohydrazide':
        prefixes = [(str(i), text) for i, nitrogen in enumerate(amino, 1)
                    for text in ligands[nitrogen]]
        name = join_prefix_parent(format_prefixes(prefixes, hierarchical_enclosures=True),
                                  'hydrazinecarbonitrile')
        configuration = ''
    else:
        options = []
        for a, b in (amino, amino[::-1]):
            prefixes = [(loc, text) for nitrogen, loc in ((a, 'N'), (b, "N'"), (imino, "N''"))
                        for text in ligands[nitrogen]]
            ordered = sorted(prefixes, key=lambda p: (prefix_alphabetization_key(p[1]), p[0]))
            options.append((tuple(loc for loc, _ in ordered),
                            format_prefixes(prefixes, hierarchical_enclosures=True)))
        name = min(options)[1] + 'guanidine'
        configuration = _local_configuration(mol, Chem, group.center, imino)
        if configuration is None:
            return None
    if configuration:
        name = f'({configuration})-' + name
        _record_configuration(mol, group.center, imino, name, configuration, core)
    return _decision(group, name, heavy, (), core, roles)


def _amidine_parent(mol, Chem, group, core, imino, amino, ligands, owned, roles, heavy):
    from .branch_rules import parent_chain_name
    from .formatting import complete_parent_name
    from .mapped_ring_parents import records
    from .parent_selection import compare_parents, graph_parent_facts, parent_frontier
    from .parent_rules import _ester_acyl_parent_paths

    n_prefixes = [(loc, text) for nitrogen, loc in ((amino, 'N'), (imino, "N'"))
                  for text in ligands[nitrogen]]
    configuration = _local_configuration(mol, Chem, group.center, imino)
    if configuration is None:
        return None
    if group.owner >= 0 and mol.GetAtomWithIdx(group.owner).IsInRing():
        rows = records(mol, Chem, scope=heavy - owned, attachment=group.owner,
                       general_branches=True)
        choices = []
        for row in rows:
            component = row.construction
            if component is None:
                continue
            locant = dict(row.atom_locants)[group.owner]
            if row.added_h_locants:
                continue
            suffix_locants = (locant,) if row.parent != 'benzene' or component.prefixes else ()
            ring = attach_suffix(row.parent, 'carboximidamide', suffix_locants)
            decorated = decorate_component(mol, Chem, MappedComponent(
                ring, component.atom_locants, component.atoms,
                tuple(component.prefixes) + tuple(n_prefixes),
                component.embedded_atoms | owned,
                hydro_prefix=component.hydro_prefix, nitrogen_site_locants=True))
            if decorated:
                choices.append((decorated, row.atom_locants))
        if len({name for name, _ in choices}) != 1:
            return None
        name, mapping = choices[0]
        descriptor = configuration
    else:
        paths = [(group.center,)] if group.owner < 0 else [
            (group.center, *path) for path in
            _ester_acyl_parent_paths(mol, Chem, group.center, group.owner)]
        candidates = []
        for path in paths:
            mapping = {idx: str(i) for i, idx in enumerate(path, 1)}
            if (set(path) - {group.center}) & owned:
                continue
            local_owned = owned | set(path)
            prefixes = list(n_prefixes)
            for idx in path:
                for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root = atom.GetIdx()
                    if root in local_owned or atom.GetAtomicNum() == 1:
                        continue
                    text, atoms = mapped_general_branch(mol, Chem, root, idx)
                    if not text or atoms & local_owned:
                        break
                    local_owned.update(atoms)
                    prefixes.append((mapping[idx], text))
                else:
                    continue
                break
            if local_owned != heavy:
                continue
            parent = attach_suffix(complete_parent_name(parent_chain_name(mol, path)), 'imidamide')
            bindings = dict(mapping)
            if configuration:
                bindings[imino] = "N'"
            skeletal = frozenset(bindings)
            text = decorate_component(mol, Chem, MappedComponent(
                parent, tuple(bindings.items()), skeletal, tuple(prefixes),
                heavy - skeletal, nitrogen_site_locants=True))
            if text:
                facts = graph_parent_facts(mol, path, mapping, principal_class=group.seniority,
                    suffix_centers=(group.center,), suffix_locants=('1',), prefixes=prefixes, name=text,
                    allow_annotations=True)
                candidates.append((text, tuple(mapping.items()), facts))
        best = parent_frontier(candidates, lambda row: row[2])
        if (len({row[0] for row in best}) != 1 or any(
                compare_parents(best[0][2], row[2]).preference not in {-1, 0} for row in candidates
                if row[0] != best[0][0])):
            return None
        name, mapping, _facts = best[0]
        descriptor = ''  # The mapped terminal C=N was decorated with its parent locant.
    if descriptor:
        name = f'({descriptor})-' + name
    if configuration:
        _record_configuration(mol, group.center, imino, name, configuration, core)
    return _decision(group, name, heavy, mapping, core, roles)


def _decision(group, name, heavy, mapping, core, roles):
    rule = {'cyanamide': 'P-41/P-66.1.6.2', 'cyanohydrazide': 'P-41/P-66.3.5.1'}.get(
        group.kind, 'P-41/P-42/P-66.4.1')
    residue = heavy - core - frozenset().union(*(atoms for _, atoms, _ in roles))
    roles = (('characteristic-unit', core, ()), *roles)
    if residue:
        roles += (('parent-and-substituents', residue,
                   tuple((idx, loc) for idx, loc in mapping if idx in residue)),)
    return MethodDecision(name, 'substitutive', frozenset({group.center}), heavy, rule,
                          mapping, component_roles=roles, preference_issues=(SelectionLimitation(
                              'nitrogen-acid-derivative', rule,
                              'The principal group and its substituents are mapped.'),))
