"""Carbon-chain prefix parents for substituted acyl and alkanoate groups."""


from ..molecule_iteration import iter_atoms
from ..formatting import complete_parent_name, format_prefixes, join_prefix_parent
from ..molecule_cache import mol_cache_get, mol_cache_set


def _ordinary_neutral_graph(mol):
    return all(
        not (a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons())
        for a in iter_atoms(mol)
    )


def ring_acid_derivative_parent(record, attachment, suffix):
    """Cite a ring acid derivative on the numbered hydride."""
    component = record.construction
    locant = dict(component.atom_locants).get(attachment)
    if not locant:
        return ""
    if record.parent == "benzene" and component.parent == "phenyl" and locant == "1":
        return {"carbonyl": "benzoyl", "carboxamido": "benzamido"}[suffix]
    tail = f"-{locant}-yl"
    if not component.parent.endswith(tail):
        return ""
    stem = component.parent[:-len(tail)]
    if record.parent.endswith("e") and stem.endswith(record.parent[:-1]):
        stem += "e"
    if not stem.endswith(record.parent):
        return ""
    return f"{stem}-{locant}-{suffix}"


def _configured_aziridine_carbonyl(mol, Chem, core, scope, ligand, carbonyl):
    """Map a saturated aziridine whose cage nitrogen has assigned CIP."""
    from ..component_mapping import MappedComponent, decorate_component, mapped_general_branch
    from ..stereochemistry import assigned_atom_stereochemistry

    if len(core) != 3 or mol.GetStereoGroups():
        return ''
    nitrogens = [i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    if len(nitrogens) != 1:
        return ''
    nitrogen = nitrogens[0]
    atom = mol.GetAtomWithIdx(nitrogen)
    if (not atom.GetChiralTag() or atom.GetTotalValence() != 3
            or dict(assigned_atom_stereochemistry(mol, Chem)).get(nitrogen) not in {'R', 'S'}
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in core - {nitrogen})
            or any(b.GetBondTypeAsDouble() != 1 for i in core
                   for b in mol.GetAtomWithIdx(i).GetBonds())):
        return ''
    mapped = ((nitrogen, '1'), (ligand, '2'), (next(iter(core - {nitrogen, ligand})), '3'))
    prefixes, occupied = [], set(core)
    for host, locant in mapped:
        for neighbor in mol.GetAtomWithIdx(host).GetNeighbors():
            root = neighbor.GetIdx()
            if root in core or root == carbonyl or neighbor.GetAtomicNum() == 1:
                continue
            word, owned = mapped_general_branch(mol, Chem, root, host)
            if not word or not owned <= scope or owned & occupied:
                return ''
            prefixes.append((locant, word))
            occupied.update(owned)
    if occupied != scope:
        return ''
    return decorate_component(mol, Chem, MappedComponent(
        'aziridine-2-carbonyl', mapped, core, tuple(prefixes), scope - core,
        systematic_parent=True))


def _carbon_paths(mol, start, blocked):
    """Enumerate bounded, saturated carbon chains rooted at the group carbon."""
    paths = []
    stack = [(start,)]
    while stack:
        path = stack.pop()
        if len(path) > 40:
            return []
        neighbors = [
            a.GetIdx()
            for a in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
            if a.GetIdx() not in blocked
            and a.GetIdx() not in path
            and a.GetAtomicNum() == 6
            and not a.IsInRing()
            and not (a.GetFormalCharge() or a.GetIsotope())
        ]
        if any(
            mol.GetBondBetweenAtoms(path[-1], idx).GetBondTypeAsDouble() != 1
            for idx in neighbors
        ):
            return []
        if neighbors:
            stack.extend(path + (idx,) for idx in neighbors)
        else:
            paths.append(path)
        if len(paths) + len(stack) > 64:
            return []
    return paths


def _path_name(mol, path, blocked, ending):
    from ..acyclic.rules import _path_substituent_prefixes
    from ..branch_rules import parent_chain_name
    from ..stereochemistry import assigned_atom_stereochemistry
    from rdkit import Chem

    prefixes = _path_substituent_prefixes(
        mol, path, skip_atoms=blocked, skip_plain_hydrogen=True
    )
    if prefixes is None:
        return ""
    prefixes = [
        (loc, "(propan-2-yl)oxy" if text == "isopropoxy" else text)
        for loc, text in prefixes
    ]
    parent = complete_parent_name(parent_chain_name(mol, path))
    if not parent.endswith("ane"):
        return ""
    if ending == "oyl" and len(path) == 2:
        name = format_prefixes(prefixes, omit_locants=True) + "acetyl"
    else:
        name = join_prefix_parent(format_prefixes(prefixes), parent[:-1] + ending)
    locants = {idx: i for i, idx in enumerate(path, 1)}
    stereo = ",".join(
        f"{locants[idx]}{descriptor}"
        for idx, descriptor in sorted(
            assigned_atom_stereochemistry(mol, Chem),
            key=lambda item: locants.get(item[0], 999),
        )
        if idx in locants
    )
    return f"({stereo})-{name}" if stereo else name


def carboxyalkanoyl_prefix(mol, Chem, carbonyl, nitrogen):
    """An acyl arm with a terminal subordinate carboxy group on a linear chain."""
    from ..branch_rules import collect_subtree
    from ..constants import ROOTS
    from ..functional_inventory import _neutral_carboxyl_query

    if mol.GetAtomWithIdx(nitrogen).GetAtomicNum() != 7:
        return ""
    acids = mol.GetSubstructMatches(_neutral_carboxyl_query(Chem), maxMatches=3)
    if len(acids) != 2:
        return ""
    scope = frozenset(collect_subtree(mol, carbonyl, nitrogen))
    local = [match for match in acids if set(match) <= scope]
    if len(local) != 1 or not any(set(match).isdisjoint(scope) for match in acids):
        return ""
    owner, acid, oxo, hydroxyl = local[0]
    if carbonyl == owner:
        return ""
    path = tuple(Chem.GetShortestPath(mol, carbonyl, owner))
    if not 2 <= len(path) <= 8 or nitrogen in scope:
        return ""
    oxygen = [b.GetOtherAtomIdx(carbonyl) for b in mol.GetAtomWithIdx(carbonyl).GetBonds()
              if b.GetBondTypeAsDouble() == 2
              and b.GetOtherAtom(mol.GetAtomWithIdx(carbonyl)).GetAtomicNum() == 8]
    if len(oxygen) != 1 or scope != {*path, *oxygen, acid, oxo, hydroxyl}:
        return ""
    if any(a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
           or a.GetChiralTag() or a.IsInRing() or a.HasQuery()
           for a in (mol.GetAtomWithIdx(i) for i in scope)):
        return ""
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path)
            or any(mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE
                   for i, j in zip(path, path[1:]))
            or mol.GetAtomWithIdx(oxygen[0]).GetDegree() != 1
            or mol.GetAtomWithIdx(carbonyl).GetDegree() != 3
            or any(mol.GetAtomWithIdx(i).GetTotalNumHs() != 2
                   or mol.GetAtomWithIdx(i).GetDegree() != 2 for i in path[1:])):
        return ""
    length = len(path)
    return '2-carboxyacetyl' if length == 2 else f'{length}-carboxy{ROOTS[length]}anoyl'


def azidoalkanoyl_prefix(mol, Chem, carbonyl, nitrogen, scope):
    """Name an acyl arm whose charge separation belongs to one terminal azide."""
    from ..constants import ROOTS
    from ..neutral_charge_components import _azido_at_host

    if (not 6 <= len(scope) <= 15 or nitrogen in scope
            or mol.GetAtomWithIdx(nitrogen).GetAtomicNum() != 7
            or mol.GetBondBetweenAtoms(carbonyl, nitrogen).GetBondType() != Chem.BondType.SINGLE):
        return None
    atom = mol.GetAtomWithIdx(carbonyl)
    if atom.IsInRing() or atom.GetDegree() != 3:
        return None
    ports = {b.GetOtherAtom(atom).GetAtomicNum(): b for b in atom.GetBonds()
             if b.GetOtherAtomIdx(carbonyl) != nitrogen}
    if (ports.keys() != {6, 8} or ports[6].GetBondTypeAsDouble() != 1
            or ports[8].GetBondTypeAsDouble() != 2):
        return None
    oxygen = ports[8].GetOtherAtom(atom)
    if oxygen.GetDegree() != 1:
        return None
    path, previous = [carbonyl], carbonyl
    atom = ports[6].GetOtherAtom(atom)
    while atom.GetAtomicNum() == 6:
        if atom.IsInRing() or atom.GetDegree() != 2 or atom.GetTotalNumHs() != 2:
            return None
        idx = atom.GetIdx()
        path.append(idx)
        following = next(a for a in atom.GetNeighbors() if a.GetIdx() != previous)
        if mol.GetBondBetweenAtoms(idx, following.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
            return None
        atom, previous = following, idx
    proof = _azido_at_host(mol, previous, atom.GetIdx())
    core = {*path, oxygen.GetIdx()}
    if proof is None or scope != core | set(proof.atoms) or any(
            a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetChiralTag() or a.HasQuery() for a in (mol.GetAtomWithIdx(i) for i in core)):
        return None
    length = len(path)
    return ('2-azidoacetyl' if length == 2 else f'{length}-azido{ROOTS[length]}anoyl'), scope


def n_attached_aminoacyl_prefix(mol, carbonyl_idx, parent_idx):
    """Name N-substituted aminoacyl branches on cyclic nitrogens as acyls.

    Free aminoacyl groups retain their existing biochemical vocabulary. Ionic,
    isotopic and unsaturated chains retain the specialized branch routes.
    """
    key = ("preferred_aminoacyl", carbonyl_idx, parent_idx)
    cached = mol_cache_get(mol, key)
    if cached is not None:
        return cached
    result = ""
    parent = mol.GetAtomWithIdx(parent_idx)
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if (
        _ordinary_neutral_graph(mol)
        and parent.GetAtomicNum() == 7
        and parent.IsInRing()
        and carbonyl.GetAtomicNum() == 6
    ):
        double_o = [
            a.GetIdx()
            for a in carbonyl.GetNeighbors()
            if a.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(carbonyl_idx, a.GetIdx()).GetBondTypeAsDouble()
            == 2
        ]
        distal = [
            a
            for a in carbonyl.GetNeighbors()
            if a.GetIdx() != parent_idx and a.GetIdx() not in double_o
        ]
        if len(double_o) == 1 and len(distal) == 1 and distal[0].GetAtomicNum() == 6:
            alpha = distal[0]
            amino = [
                a
                for a in alpha.GetNeighbors()
                if a.GetAtomicNum() == 7
                and a.GetIdx() != parent_idx
                and not a.IsInRing()
                and not a.GetFormalCharge()
                and any(
                    b.GetAtomicNum() > 1 and b.GetIdx() != alpha.GetIdx()
                    for b in a.GetNeighbors()
                )
            ]
            if len(amino) == 1:
                paths = _carbon_paths(mol, carbonyl_idx, {parent_idx})
                names = [
                    (len(p), _path_name(mol, p, {parent_idx, *double_o}, "oyl"))
                    for p in paths
                ]
                names = [(n, s) for n, s in names if s]
                if names:
                    result = min(names, key=lambda r: (-r[0], r[1]))[1]
    return mol_cache_set(mol, key, result)


def n_attached_heterocyclylacetyl_prefix(mol, carbonyl_idx, parent_idx):
    """Include the methylene in a ring-bound, heterocyclyl-substituted acetyl.

    P-57.4 selects the largest significant acyl component including the free
    valence; P-34.2.1.1 permits substitution on retained acetyl.  Restrict this
    path to a proven two-carbon chain and let the mapped path writer name the
    attached heterocycle and its substituents.
    """
    key = ("preferred_heterocyclylacetyl", carbonyl_idx, parent_idx)
    cached = mol_cache_get(mol, key)
    if cached is not None:
        return cached
    result = ""
    parent = mol.GetAtomWithIdx(parent_idx)
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if (
        _ordinary_neutral_graph(mol)
        and parent.GetAtomicNum() == 7
        and parent.IsInRing()
        and carbonyl.GetAtomicNum() == 6
        and not carbonyl.IsInRing()
    ):
        double_o = [
            a.GetIdx() for a in carbonyl.GetNeighbors()
            if a.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(carbonyl_idx, a.GetIdx()).GetBondTypeAsDouble() == 2
        ]
        distal = [
            a for a in carbonyl.GetNeighbors()
            if a.GetAtomicNum() > 1 and a.GetIdx() != parent_idx and a.GetIdx() not in double_o
        ]
        if len(double_o) == 1 and len(distal) == 1:
            alpha = distal[0]
            neighbors = [
                a for a in alpha.GetNeighbors()
                if a.GetAtomicNum() > 1 and a.GetIdx() != carbonyl_idx
            ]
            if (
                alpha.GetAtomicNum() == 6
                and not alpha.IsInRing()
                and alpha.GetTotalNumHs(includeNeighbors=True) == 2
                and mol.GetBondBetweenAtoms(carbonyl_idx, alpha.GetIdx()).GetBondTypeAsDouble() == 1
                and len(neighbors) == 1
                and neighbors[0].GetAtomicNum() in {7, 8, 16}
                and neighbors[0].IsInRing()
                and mol.GetBondBetweenAtoms(alpha.GetIdx(), neighbors[0].GetIdx()).GetBondTypeAsDouble() == 1
                and _carbon_paths(mol, carbonyl_idx, {parent_idx}) == [(carbonyl_idx, alpha.GetIdx())]
            ):
                result = _path_name(mol, (carbonyl_idx, alpha.GetIdx()), {parent_idx, *double_o}, "oyl")
    return mol_cache_set(mol, key, result)


def ring_n_attached_ester_ethyl_prefix(mol, root_idx, parent_idx):
    """Name ring N-CH2-C(=O)-OR as an alkoxy-oxoethyl group.

    P-57.4 chooses the larger carbon component and P-65.6.3.2.3
    expresses a subordinate ester by alkoxy and oxo prefixes. Keep the
    route to a fully verified, unlabelled two-carbon ester component.
    """
    key = ("preferred_ring_n_ester_ethyl", root_idx, parent_idx)
    cached = mol_cache_get(mol, key)
    if cached is not None:
        return cached
    result = ""
    parent = mol.GetAtomWithIdx(parent_idx)
    root = mol.GetAtomWithIdx(root_idx)
    bond = mol.GetBondBetweenAtoms(root_idx, parent_idx)
    if not (
        parent.GetAtomicNum() == 7 and parent.IsInRing()
        and root.GetAtomicNum() == 6 and not root.IsInRing()
        and root.GetTotalNumHs(includeNeighbors=True) == 2
        and bond is not None and bond.GetBondTypeAsDouble() == 1
        and _ordinary_neutral_graph(mol)
    ):
        return mol_cache_set(mol, key, result)
    children = [a for a in root.GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() != parent_idx]
    if len(children) != 1 or children[0].GetAtomicNum() != 6:
        return mol_cache_set(mol, key, result)
    carbonyl = children[0]
    carbonyl_idx = carbonyl.GetIdx()
    carbonyl_bond = mol.GetBondBetweenAtoms(root_idx, carbonyl_idx)
    oxygen_neighbors = [a for a in carbonyl.GetNeighbors() if a.GetIdx() != root_idx and a.GetAtomicNum() > 1]
    if (
        carbonyl.IsInRing()
        or carbonyl_bond is None or carbonyl_bond.GetBondTypeAsDouble() != 1
        or len(oxygen_neighbors) != 2
        or any(a.GetAtomicNum() != 8 for a in oxygen_neighbors)
    ):
        return mol_cache_set(mol, key, result)
    by_order = {
        mol.GetBondBetweenAtoms(carbonyl_idx, a.GetIdx()).GetBondTypeAsDouble(): a
        for a in oxygen_neighbors
    }
    if set(by_order) != {1.0, 2.0}:
        return mol_cache_set(mol, key, result)
    single_o, double_o = by_order[1.0], by_order[2.0]
    if (
        single_o.IsInRing() or single_o.GetTotalNumHs(includeNeighbors=True)
        or len([a for a in single_o.GetNeighbors() if a.GetAtomicNum() > 1]) != 2
        or len([a for a in double_o.GetNeighbors() if a.GetAtomicNum() > 1]) != 1
    ):
        return mol_cache_set(mol, key, result)
    from ..acyclic.rules import _path_substituent_prefixes
    prefixes = _path_substituent_prefixes(
        mol, (root_idx, carbonyl_idx), skip_atoms={parent_idx}, skip_plain_hydrogen=True
    )
    if prefixes is not None and len(prefixes) == 2:
        result = format_prefixes(prefixes) + "ethyl"
    return mol_cache_set(mol, key, result)


def ring_carbonyl_prefix(mol, carbonyl_idx, parent_idx, *, allow_azido=False):
    """Use the acid-derived prefix on a completely mapped ring acyl group.

    P-65.1.7.4.2 derives the prefix from the ring carboxylic acid. The
    ring's hydride, numbering and every substituent remain tied to one
    complete original-graph record.
    """
    from rdkit import Chem
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetAtomicNum() != 6 or carbonyl.IsInRing():
        return ''
    neighbors = [a for a in carbonyl.GetNeighbors() if a.GetIdx() != parent_idx and a.GetAtomicNum() > 1]
    oxygens = [a for a in neighbors if a.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(carbonyl_idx, a.GetIdx()).GetBondTypeAsDouble() == 2]
    rings = [a for a in neighbors if a.GetAtomicNum() == 6 and a.IsInRing()
             and mol.GetBondBetweenAtoms(carbonyl_idx, a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(neighbors) != 2 or len(oxygens) != 1 or len(rings) != 1
            or carbonyl.GetIsotope() or oxygens[0].GetIsotope()):
        return ''
    from ..component_mapping import decorate_component, has_annotations
    from ..mapped_ring_parents import records, ring_systems
    from .functional_groups import collect_subtree
    scope = frozenset(collect_subtree(mol, carbonyl_idx, parent_idx))
    ligand_idx = rings[0].GetIdx()
    ligand_atoms = frozenset(collect_subtree(mol, ligand_idx, carbonyl_idx))
    if parent_idx in scope or scope != ligand_atoms | {carbonyl_idx, oxygens[0].GetIdx()}:
        return ''
    charged = set()
    for i in scope:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetNumRadicalElectrons():
            return ''
        if atom.GetFormalCharge():
            if not allow_azido:
                return ''
            charged.add(i)
    if charged:
        from ..neutral_charge_components import neutral_charge_role_proofs

        proofs = tuple(proof for proof in neutral_charge_role_proofs(mol)
                       if proof.kind == 'azido'
                       and {proof.attachment_atom, *proof.atoms} <= ligand_atoms)
        if charged != {i for proof in proofs for i in proof.charge_atom_ids}:
            return ''
    core = next((system for system in ring_systems(mol) if ligand_idx in system), None)
    if core is None or not 3 <= len(core) <= 16 or not core <= ligand_atoms:
        return ''
    carbocycle = all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in core)
    if carbocycle:
        if len(core) > 8 or has_annotations(mol, scope):
            return ''
        for i in core:
            atom = mol.GetAtomWithIdx(i)
            annular = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(i) in core]
            if len(annular) != 2 or any(b.GetBondTypeAsDouble() != 1 for b in annular):
                return ''  # Benzoyl and other hydrocarbon acyls have established routes.
    candidates = records(mol, Chem, scope=ligand_atoms, attachment=ligand_idx, core_filter=core,
                         general_branches=bool(charged))
    if len(candidates) != 1:
        return '' if charged else _configured_aziridine_carbonyl(
            mol, Chem, core, ligand_atoms, ligand_idx, carbonyl_idx)
    record = candidates[0]
    component = record.construction
    if (component is None or not record.hydrogen_budget_proved or not record.annotation_coverage
            or frozenset(record.parent_atoms) != core
            or component.atoms | component.embedded_atoms != ligand_atoms):
        return ''
    if charged:
        from ..component_mapping import _general_branch_stereo_is_cited

        if (any(not any(set(proof.atoms) <= owned and proof.attachment_atom in owned | {host}
                        for _, host, owned, _ in component.prefix_branch_ownership)
                for proof in proofs)
                or not _general_branch_stereo_is_cited(mol, Chem, ligand_atoms, record.name)):
            return ''
    if carbocycle:
        from ..constants import ROOTS
        parent = 'cyclo' + ROOTS[len(core)] + 'ane'
        if record.parent != parent or dict(component.atom_locants).get(ligand_idx) != '1':
            return ''
        text = parent + 'carbonyl'
    else:
        text = ring_acid_derivative_parent(record, ligand_idx, 'carbonyl')
        if not text:
            return ''
    from dataclasses import replace
    return decorate_component(mol, Chem, replace(component, parent=text))


def alkanoate_amino_ligand(mol, root_idx, nitrogen_idx):
    """Include the ester carbon in the carbon-chain parent of an amino ligand."""
    key = ("preferred_alkanoate_ligand", root_idx, nitrogen_idx)
    cached = mol_cache_get(mol, key)
    if cached is not None:
        return cached
    result = ""
    root = mol.GetAtomWithIdx(root_idx)
    if (
        _ordinary_neutral_graph(mol)
        and root.GetAtomicNum() == 6
        and not root.IsInRing()
    ):
        for carbon in root.GetNeighbors():
            if (
                carbon.GetAtomicNum() != 6
                or carbon.IsInRing()
                or carbon.GetFormalCharge()
            ):
                continue
            oxygens = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8]
            if len(oxygens) != 2:
                continue
            if sorted(
                mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), a.GetIdx()
                ).GetBondTypeAsDouble()
                for a in oxygens
            ) != [1, 2]:
                continue
            single_o = next(
                a
                for a in oxygens
                if mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), a.GetIdx()
                ).GetBondTypeAsDouble()
                == 1
            )
            if single_o.GetDegree() != 2 or single_o.GetFormalCharge():
                continue
            paths = _carbon_paths(mol, carbon.GetIdx(), {nitrogen_idx})
            names = [
                (len(p), _path_name(mol, p, {nitrogen_idx}, "-2-yl"))
                for p in paths
                if len(p) > 1 and p[1] == root_idx
            ]
            names = [(n, s) for n, s in names if s]
            if names:
                result = min(names, key=lambda r: (-r[0], r[1]))[1]
                break
    return mol_cache_set(mol, key, result)
