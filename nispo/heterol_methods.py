"""P-63.1.3 hydroxyl suffixes on saturated heterocyclic parents."""

from types import SimpleNamespace

from .naming_methods import MethodDecision


def group14_heterol_decision(mol, Chem):
    """Name neutral, mononuclear Group 14 hydroxy hydrides by -ol.

    Hydrocarbyl ligands are carbon-only and all oxygens are hydroxyl groups;
    this keeps more senior characteristic groups and acid exceptions out.
    """
    parents = {14: 'silane', 32: 'germane', 50: 'stannane', 82: 'plumbane'}
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 8, *parents} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in parents]
    if len(centers) != 1 or centers[0].IsInRing() or centers[0].GetTotalValence() != 4:
        return None
    center = centers[0]
    cid = center.GetIdx()
    hydroxyls, roots = [], []
    for atom in center.GetNeighbors():
        if mol.GetBondBetweenAtoms(cid, atom.GetIdx()).GetBondTypeAsDouble() != 1:
            return None
        if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                and atom.GetTotalNumHs(includeNeighbors=True) == 1):
            hydroxyls.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 6:
            roots.append(atom.GetIdx())
        else:
            return None
    if not 1 <= len(hydroxyls) <= 4:
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    from .formatting import attach_suffix, multiplied_suffix

    core = frozenset((cid, *hydroxyls))
    seen, branches, units = set(core), [], []
    for root in roots:
        atoms = _carbon_component(mol, root, cid)
        if atoms is None or atoms & seen:
            return None
        text, owned = mapped_branch(mol, Chem, root, cid)
        if not text or frozenset(owned) != atoms:
            return None
        path = all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                   for i in atoms)
        units.append(PrefixUnit(text, 'simple' if text in {'phenyl', 'tert-butyl'} or path else 'compound'))
        branches.append(atoms)
        seen.update(atoms)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy:
        return None
    prefix = mononuclear_prefix_sequence(units) if units else None
    if units and prefix is None:
        return None
    silicic_acid = center.GetAtomicNum() == 14 and len(hydroxyls) == 4
    ending = ('silicic acid' if silicic_acid else
              attach_suffix(parents[center.GetAtomicNum()], multiplied_suffix(len(hydroxyls), 'ol')))
    name = (prefix.text if prefix else '') + ending
    roles = [('heterohydride-and-ols', core, ())]
    roles.extend(('organyl-substituent', atoms, ()) for atoms in branches)
    return MethodDecision(name, 'retained-parent' if silicic_acid else 'substitutive',
                          frozenset({cid}), heavy,
                          'P-63.1.3/P-67.1.1/P-41/P-14.3.4.2', (), component_roles=tuple(roles))


def pnictogen_ous_acid_decision(mol, Chem):
    """Keep P/As/Sb oxoacid parent names ahead of a heterol suffix.

    P-67.1.1.1-.2 retains the trivalent hydroxy acids and permits their
    central H atoms to be replaced by carbon-only organyl groups.
    """
    parents = {
        15: {1: 'phosphinous acid', 2: 'phosphonous acid', 3: 'phosphorous acid'},
        33: {1: 'arsinous acid', 2: 'arsonous acid', 3: 'arsorous acid'},
        51: {1: 'stibinous acid', 2: 'stibonous acid', 3: 'stiborous acid'},
    }
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 8, *parents} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() in parents]
    if len(centers) != 1 or centers[0].IsInRing() or centers[0].GetTotalValence() != 3:
        return None
    center = centers[0]
    cid = center.GetIdx()
    hydroxy, roots = [], []
    for atom in center.GetNeighbors():
        if mol.GetBondBetweenAtoms(cid, atom.GetIdx()).GetBondTypeAsDouble() != 1:
            return None
        if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                and atom.GetTotalNumHs(includeNeighbors=True) == 1):
            hydroxy.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 6:
            roots.append(atom.GetIdx())
        else:
            return None
    if len(hydroxy) not in {1, 2, 3}:
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    core = frozenset((cid, *hydroxy))
    seen, branches, units = set(core), [], []
    for root in roots:
        atoms = _carbon_component(mol, root, cid)
        if atoms is None or atoms & seen:
            return None
        text, owned = mapped_branch(mol, Chem, root, cid)
        if not text or frozenset(owned) != atoms:
            return None
        units.append(PrefixUnit(text, 'simple' if text in {'phenyl', 'tert-butyl'}
                                or all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                                       for i in atoms) else 'compound'))
        branches.append(atoms)
        seen.update(atoms)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy:
        return None
    prefix = mononuclear_prefix_sequence(units) if units else None
    if units and prefix is None:
        return None
    name = (prefix.text if prefix else '') + parents[center.GetAtomicNum()][len(hydroxy)]
    roles = [('acid-core', core, ())]
    roles.extend(('organyl-substituent', atoms, ()) for atoms in branches)
    return MethodDecision(name, 'retained-parent', frozenset({cid}), heavy,
                          'P-63.1.3/P-67.1.1.1/P-67.1.1.2', (), component_roles=tuple(roles))


def group14_cyclic_heterol_decision(mol, Chem):
    """Name single saturated Group 14/carbon rings with an annular E-OH."""
    stems = {14: 'sil', 32: 'germ', 50: 'stann', 82: 'plumb'}
    roots = {3: 'irane', 4: 'etane', 5: 'olane', 6: 'inane', 7: 'epane', 8: 'ocane'}
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings() != 1
            or any(a.GetAtomicNum() not in {6, 8, 9, 17, 35, 53, *stems}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    from .mapped_ring_parents import ring_systems
    from .component_mapping import mapped_branch
    from .formatting import (attach_suffix, format_prefixes, join_prefix_parent,
                             multiplied_suffix, prefix_alphabetization_key)

    rings = ring_systems(mol)
    if len(rings) != 1:
        return None
    core = rings[0]
    centers = [i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() in stems]
    if (len(core) not in roots or len(centers) != 1
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core if i not in centers)):
        return None
    center = centers[0]
    if mol.GetAtomWithIdx(center).GetTotalValence() != 4:
        return None
    neighbors = {}
    for idx in core:
        neighbors[idx] = [a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors() if a.GetIdx() in core]
        if len(neighbors[idx]) != 2 or any(
                mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1 for other in neighbors[idx]):
            return None
    hydroxyls = []
    for idx in core:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            if (atom.GetIdx() not in core and atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                    and atom.GetTotalNumHs(includeNeighbors=True) == 1
                    and mol.GetBondBetweenAtoms(idx, atom.GetIdx()).GetBondTypeAsDouble() == 1):
                hydroxyls.append((idx, atom.GetIdx()))
    if not any(host == center for host, _ in hydroxyls):
        return None
    suffix_atoms = frozenset(oxygen for _host, oxygen in hydroxyls)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    parent = stems[mol.GetAtomWithIdx(center).GetAtomicNum()] + roots[len(core)]
    rows = []
    for first in neighbors[center]:
        seq, prev, current = [center], center, first
        while current != center and len(seq) <= len(core):
            seq.append(current)
            nxt = [item for item in neighbors[current] if item != prev]
            if len(nxt) != 1:
                break
            prev, current = current, nxt[0]
        if current != center or frozenset(seq) != core or len(seq) != len(core):
            continue
        mapping = {idx: str(i + 1) for i, idx in enumerate(seq)}
        suffix_locs = tuple(sorted(int(mapping[host]) for host, _ in hydroxyls))
        prefix_rows, branches, roles = [], set(), []
        valid = True
        for host in seq:
            for atom in mol.GetAtomWithIdx(host).GetNeighbors():
                root = atom.GetIdx()
                if root in core or root in suffix_atoms:
                    continue
                text, atoms = mapped_branch(mol, Chem, root, host)
                if (not text or not atoms or atoms & (core | suffix_atoms | branches)
                        or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53}
                               for i in atoms)):
                    valid = False
                    break
                prefix_rows.append((mapping[host], text))
                branches.update(atoms)
                roles.append(('substituent', frozenset(atoms), ()))
            if not valid:
                break
        if not valid or core | suffix_atoms | branches != heavy:
            continue
        suffix = multiplied_suffix(len(hydroxyls), 'ol')
        if not suffix:
            continue
        ending = attach_suffix(parent, suffix, suffix_locs)
        name = join_prefix_parent(format_prefixes(prefix_rows), ending) if prefix_rows else ending
        text_order = sorted({text for _loc, text in prefix_rows}, key=prefix_alphabetization_key)
        score = (suffix_locs, tuple(sorted(int(loc) for loc, _text in prefix_rows)),
                 tuple(tuple(sorted(int(loc) for loc, value in prefix_rows if value == text))
                       for text in text_order))
        rows.append((score, name, mapping, roles))
    if not rows:
        return None
    best_score = min(row[0] for row in rows)
    best = [row for row in rows if row[0] == best_score]
    if len({row[1] for row in best}) != 1:
        return None
    _score, name, mapping, branches = best[0]
    roles = [('ring-parent', core, tuple(mapping.items())), ('hydroxy-suffixes', suffix_atoms, ())]
    roles.extend(branches)
    return MethodDecision(name, 'substitutive', frozenset(host for host, _ in hydroxyls), heavy,
                          'P-63.1.3/P-41/P-44/P-45', tuple(mapping.items()),
                          component_roles=tuple(roles))


def cyclic_heterol_decision(mol, Chem):
    """Own a saturated monocycle's N-OH and optional annular C-OH groups.

    Other branches are limited to carbon-only groups and terminal halogens;
    this proves absence of a more senior characteristic group. Isotopic,
    stereo, fused, and unsaturated-ring cases retain their existing routes.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings() != 1
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 16, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    from .mapped_ring_parents import parent_facts, ring_systems, skeletons
    from .component_mapping import MappedComponent, mapped_branch
    from .formatting import attach_suffix, format_prefixes, join_prefix_parent, multiplied_suffix
    from .parent_selection import parent_frontier

    rings = ring_systems(mol)
    if len(rings) != 1:
        return None
    core = rings[0]
    if not 3 <= len(core) <= 8 or any(
            mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
            for a in core for b in core if a < b and mol.GetBondBetweenAtoms(a, b)):
        return None
    hydroxyls = []
    for idx in core:
        host = mol.GetAtomWithIdx(idx)
        for oxygen in host.GetNeighbors():
            oidx = oxygen.GetIdx()
            if (oidx not in core and oxygen.GetAtomicNum() == 8 and oxygen.GetDegree() == 1
                    and oxygen.GetTotalNumHs(includeNeighbors=True) == 1
                    and mol.GetBondBetweenAtoms(idx, oidx).GetBondTypeAsDouble() == 1):
                if host.GetAtomicNum() not in {6, 7}:
                    return None
                hydroxyls.append((idx, oidx))
    if not hydroxyls or not any(mol.GetAtomWithIdx(host).GetAtomicNum() == 7 for host, _ in hydroxyls):
        return None
    suffix_atoms = frozenset(oxygen for _host, oxygen in hydroxyls)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    candidates = []
    for skeleton in skeletons(mol, Chem, core):
        parent, seq, locants = skeleton[:3]
        if not skeleton[5] or skeleton[7] != 'Hantzsch-Widman':
            continue
        mapping = dict(zip(seq, locants, strict=True))
        suffix_locs = tuple(sorted((mapping[host] for host, _oxygen in hydroxyls), key=int))
        prefixes, branches, ownership = [], set(), []
        valid = True
        for host in seq:
            for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
                root = neighbor.GetIdx()
                if root in core or root in suffix_atoms:
                    continue
                text, atoms = mapped_branch(mol, Chem, root, host)
                if (not text or not atoms or atoms & (core | suffix_atoms | branches)
                        or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 9, 17, 35, 53}
                               for i in atoms)):
                    valid = False
                    break
                prefixes.append((mapping[host], text))
                branches.update(atoms)
                ownership.append((root, host, frozenset(atoms), text))
            if not valid:
                break
        if not valid or core | suffix_atoms | branches != heavy:
            continue
        suffix = multiplied_suffix(len(hydroxyls), 'ol')
        if not suffix:
            continue
        ending = attach_suffix(parent, suffix, suffix_locs)
        name = join_prefix_parent(format_prefixes(prefixes), ending) if prefixes else ending
        bindings = tuple(mapping.items()) + tuple((oxygen, '') for _host, oxygen in hydroxyls)
        component = MappedComponent(ending, bindings, core | suffix_atoms, tuple(prefixes),
                                    frozenset(branches), systematic_parent=True,
                                    prefix_branch_ownership=tuple(ownership))
        record = SimpleNamespace(parent=parent, parent_atoms=tuple(seq),
            atom_locants=tuple(mapping.items()), principal_centers=frozenset(host for host, _ in hydroxyls),
            suffix_locants=suffix_locs, prefixes=tuple(prefixes), name=name,
            indicated_h_locants=(), method='heterol', construction=component)
        facts = parent_facts(mol, record, (17, 0))
        if facts is not None:
            candidates.append((record, facts, frozenset(branches), tuple(ownership)))
    frontier = parent_frontier(candidates, lambda row: row[1])
    if not frontier or len({row[0].name for row in frontier}) != 1:
        return None
    record, _facts, branches, ownership = frontier[0]
    roles = [('ring-parent', core, record.atom_locants), ('hydroxy-suffixes', suffix_atoms, ())]
    roles.extend(('substituent', atoms, ()) for _root, _host, atoms, _text in ownership)
    if sum(len(row[1]) for row in roles) != len(heavy):
        return None
    return MethodDecision(record.name, 'substitutive', record.principal_centers, heavy,
                          'P-63.1.3/P-41/P-44/P-45', record.construction.atom_locants,
                          component_roles=tuple(roles))
