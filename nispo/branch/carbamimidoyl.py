"""P-66.4.1.3.1 acyl-derived amidine substituents."""


def amidoxime_alkyl_prefix(mol, Chem, root, host, scope):
    """Carry a terminal amidoxime group through a singly rooted alkyl chain."""
    from ..amidine_methods import _amidoxime_query
    from ..branch_rules import collect_subtree
    from ..formatting import format_prefixes
    from ..constants import ROOTS

    ends = {match[0] for match in mol.GetSubstructMatches(_amidoxime_query(Chem))
            if set(match) <= scope}
    if len(ends) != 1 or host in scope:
        return None
    end = ends.pop()
    path, previous, idx = [], host, root
    while idx != end:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 2
                or atom.GetTotalNumHs() != 2 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons() or atom.GetChiralTag()):
            return None
        following = next(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous)
        if mol.GetBondBetweenAtoms(idx, following).GetBondType() != Chem.BondType.SINGLE:
            return None
        path.append(idx)
        previous, idx = idx, following
    if len(path) not in ROOTS or not path:
        return None
    if ({(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
         if n.GetIdx() not in scope} != {(root, host)}
            or frozenset(collect_subtree(mol, idx, previous)) | set(path) != scope):
        return None
    prefix = carbamimidoyl_prefix(mol, Chem, idx, previous)
    if not prefix:
        return None
    head = format_prefixes([(str(len(path)), prefix)], omit_locants=len(path) == 1,
                           hierarchical_enclosures=True)
    return head + ROOTS[len(path)] + 'yl', scope


def carbamimidoyl_prefix(mol, Chem, root, host):
    """Name a neutral C(=N)-N branch with independently mapped N ligands.

    The amino N is N and the imino N is N'. Unsupported annotations, ring
    closures, and ligand functionality are left to other producers.
    """
    carbon = mol.GetAtomWithIdx(root)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or bond is None
            or bond.GetBondTypeAsDouble() != 1 or carbon.GetFormalCharge()
            or carbon.GetIsotope() or carbon.GetNumRadicalElectrons()
            or carbon.GetTotalNumHs(includeNeighbors=True)):
        return ""
    others = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() != host]
    if len(others) != 2 or any(a.GetAtomicNum() != 7 for a in others):
        return ""
    amino = [a for a in others if mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 1]
    imino = [a for a in others if mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(amino) != 1 or len(imino) != 1:
        return ""
    from ..phosphorus_ester_methods import _carbon_component
    from ..component_mapping import mapped_branch
    from ..formatting import format_prefixes

    prefixes, owned = [], {root, host, amino[0].GetIdx(), imino[0].GetIdx()}
    for nitrogen, locant, max_ligands in ((amino[0], "N", 2), (imino[0], "N'", 1)):
        if (nitrogen.GetFormalCharge() or nitrogen.GetIsotope()
                or nitrogen.GetNumRadicalElectrons() or nitrogen.IsInRing()
                or int(nitrogen.GetChiralTag())):
            return ""
        ligands = [a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() != root]
        if len(ligands) > max_ligands:
            return ""
        for ligand in ligands:
            idx = ligand.GetIdx()
            if mol.GetBondBetweenAtoms(nitrogen.GetIdx(), idx).GetBondTypeAsDouble() != 1:
                return ""
            if (locant == "N'" and ligand.GetAtomicNum() == 8
                    and ligand.GetDegree() == 1 and ligand.GetTotalNumHs(includeNeighbors=True) == 1
                    and not (ligand.GetFormalCharge() or ligand.GetIsotope()
                             or ligand.GetNumRadicalElectrons())
                    and all(a.GetAtomicNum() != 1 or not a.GetIsotope() for a in ligand.GetNeighbors())):
                # P-66.4.4: hydroxylation is on the imino N of the amidine.
                owned.add(idx)
                prefixes.append((locant, "hydroxy"))
                continue
            atoms = _carbon_component(mol, idx, nitrogen.GetIdx())
            if atoms is None or atoms & owned:
                return ""
            text, mapped = mapped_branch(mol, Chem, idx, nitrogen.GetIdx())
            if not text or frozenset(mapped) != atoms:
                return ""
            owned.update(atoms)
            prefixes.append((locant, text))
    text = format_prefixes(prefixes)
    text += "carbamimidoyl"
    from ..stereochemistry import assigned_bond_stereochemistry
    from ..molecule_cache import mol_cache_dict
    imino_idx = imino[0].GetIdx()
    descriptor = next((d for a, b, d in assigned_bond_stereochemistry(mol, Chem)
                       if {a, b} == {root, imino_idx}), None)
    if not descriptor and mol.GetBondBetweenAtoms(root, imino_idx).GetStereo():
        return ""
    if descriptor:
        # This separately named characteristic group has one C=N unit.
        # Its configuration belongs inside the group, ahead of composition
        # with amino/piperazinyl and any independently unsaturated parent.
        text = f"({descriptor})-" + text
        mol_cache_dict(mol, 'rendered_mapped_branch_bond_stereo')[
            (root, imino_idx, text, frozenset(owned - {host}))] = descriptor
    return text
