"""Functional-class nitriles of carbonic and cyanic acid families."""

from .naming_methods import MethodDecision


def _decision(mol, name, centers, rule):
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return MethodDecision(name, 'functional-class', frozenset(centers), atoms, rule, (),
                          component_roles=(('carbon-acid-and-pseudohalide', atoms, ()),))


def _plain(mol, Chem, charges=False):
    return (mol.GetNumHeavyAtoms() <= 64 and len(Chem.GetMolFrags(mol)) == 1
            and not mol.GetStereoGroups()
            and all(not (a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) or a.HasQuery())
                    and (charges or not a.GetFormalCharge()) for a in mol.GetAtoms())
            and all(not (int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()) for b in mol.GetBonds()))


def _cyano(mol, carbon, host):
    atom = mol.GetAtomWithIdx(carbon)
    if atom.GetAtomicNum() != 6 or atom.GetDegree() != 2 or atom.GetFormalCharge():
        return None
    if mol.GetBondBetweenAtoms(carbon, host).GetBondTypeAsDouble() != 1:
        return None
    others = [a for a in atom.GetNeighbors() if a.GetIdx() != host]
    if (len(others) != 1 or others[0].GetAtomicNum() != 7 or others[0].GetDegree() != 1
            or others[0].GetFormalCharge()
            or mol.GetBondBetweenAtoms(carbon, others[0].GetIdx()).GetBondTypeAsDouble() != 3):
        return None
    return frozenset((carbon, others[0].GetIdx()))


def _polycarbonic_dicyanide(mol):
    """Prove a complete linear polycarbonic chain ending in two cyanides."""
    from .constants import MULT, ROOTS

    carbonyls = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 3 or atom.IsInRing():
            continue
        oxo = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        singles = [a for a in atom.GetNeighbors()
                   if mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) == 1 and len(singles) == 2:
            carbonyls[atom.GetIdx()] = (oxo[0].GetIdx(), singles)
    count = len(carbonyls)
    if not 2 <= count <= 20 or count not in MULT or 2 * count - 1 not in ROOTS:
        return None
    owned, links, cyanos = set(carbonyls), set(), []
    degrees = {i: 0 for i in carbonyls}
    for center, (oxygen, singles) in carbonyls.items():
        owned.add(oxygen)
        for atom in singles:
            cyano = _cyano(mol, atom.GetIdx(), center) if atom.GetAtomicNum() == 6 else None
            if cyano:
                cyanos.append((center, cyano))
                owned.update(cyano)
                continue
            if atom.GetAtomicNum() != 8 or atom.GetDegree() != 2 or atom.GetTotalNumHs():
                return None
            opposite = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != center]
            if (len(opposite) != 1 or opposite[0] not in carbonyls
                    or mol.GetBondBetweenAtoms(atom.GetIdx(), opposite[0]).GetBondTypeAsDouble() != 1):
                return None
            owned.add(atom.GetIdx())
            links.add(tuple(sorted((center, opposite[0]))))
            degrees[center] += 1
    if (len(cyanos) != 2 or len(links) != len(carbonyls) - 1
            or sorted(degrees.values()) != [1, 1] + [2] * (len(carbonyls) - 2)
            or any(degrees[center] != 1 for center, _atoms in cyanos)
            or owned != {a.GetIdx() for a in mol.GetAtoms()}):
        return None
    # Acyclicity plus the degree sequence and edge count imply one path, but
    # still verify connectivity explicitly before naming the complete graph.
    visited, stack = set(), [next(iter(carbonyls))]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(b if a == current else a for a, b in links if current in (a, b))
    if visited != set(carbonyls):
        return None
    if count <= 4:
        acid_adjective = MULT[count] + 'carbonic'
    else:
        # P-65.2 uses replacement nomenclature rather than retained carbonic
        # parent words for five or more carbonyl centers. The external two
        # C=O atoms are represented by the terminal dioic-acid operation.
        oxo = ','.join(str(i) for i in range(3, 2 * count - 2, 2))
        oxa = ','.join(str(i) for i in range(2, 2 * count - 1, 2))
        acid_adjective = (oxo + '-' + MULT[count - 2] + 'oxo-' + oxa + '-'
                          + MULT[count - 1] + 'oxa' + ROOTS[2 * count - 1] + 'anedioic')
    return _decision(mol, acid_adjective + ' dicyanide',
                     carbonyls, 'P-66.5.3.1/P-65.5.3.2/P-41')


def cyanic_pseudohalide_decision(mol, Chem):
    """P-65.5.3.3 acyl halides and pseudohalides of carbononitridic acid."""
    if not _plain(mol, Chem, charges=True):
        return None
    halides = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    from .acyl_pseudohalide_methods import _pseudohalide_at

    rows = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.GetDegree() != 2 or carbon.GetFormalCharge():
            continue
        triples = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7 and a.GetDegree() == 1
                   and a.GetFormalCharge() == 0
                   and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 3]
        triple_ids = {a.GetIdx() for a in triples}
        other = [a for a in carbon.GetNeighbors() if a.GetIdx() not in triple_ids]
        if len(triples) != 1 or len(other) != 1:
            continue
        root = other[0]
        if mol.GetBondBetweenAtoms(carbon.GetIdx(), root.GetIdx()).GetBondTypeAsDouble() != 1:
            continue
        if root.GetAtomicNum() in halides and root.GetDegree() == 1 and not root.GetFormalCharge():
            kind, owned = halides[root.GetAtomicNum()], frozenset((root.GetIdx(),))
        else:
            group = _pseudohalide_at(mol, carbon.GetIdx(), root.GetIdx())
            if group is None or group.kind == 'cyanide':
                continue  # oxalonitrile retains its own PIN
            kind, owned = group.kind, group.atoms
        core = owned | {carbon.GetIdx(), triples[0].GetIdx()}
        if core != heavy or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
            continue
        rows.append(_decision(mol, 'carbononitridic ' + kind, (carbon.GetIdx(),), 'P-65.5.3.3/P-41'))
    return rows[0] if len(rows) == 1 else None


def carbonic_cyanide_decision(mol, Chem):
    """P-66.5.3.1 carbonyl, carbamoyl and dicarbonic cyanides."""
    if not _plain(mol, Chem):
        return None
    polymer = _polycarbonic_dicyanide(mol)
    if polymer is not None:
        return polymer
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    rows = []
    for center in mol.GetAtoms():
        if center.GetAtomicNum() != 6 or center.IsInRing() or center.GetDegree() != 3:
            continue
        cidx = center.GetIdx()
        nitriles = [(n.GetIdx(), _cyano(mol, n.GetIdx(), cidx)) for n in center.GetNeighbors()
                    if n.GetAtomicNum() == 6 and _cyano(mol, n.GetIdx(), cidx)]
        doubles = [n for n in center.GetNeighbors()
                   if mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 2]
        singles = [n for n in center.GetNeighbors()
                   if mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(doubles) != 1 or len(singles) != 2:
            continue
        d = doubles[0]
        if len(nitriles) == 2:
            owned = {cidx, d.GetIdx()}
            for _root, atoms in nitriles:
                owned.update(atoms)
            if d.GetAtomicNum() == 8 and d.GetDegree() == 1:
                stem = 'carbonyl'
            elif d.GetAtomicNum() == 7 and d.GetDegree() == 1 and d.GetTotalNumHs(includeNeighbors=True) == 1:
                stem = 'carbonimidoyl'
            elif d.GetAtomicNum() == 7 and d.GetDegree() == 2:
                extra = [a for a in d.GetNeighbors() if a.GetIdx() != cidx]
                if (len(extra) != 1 or extra[0].GetAtomicNum() != 7 or extra[0].GetDegree() != 1
                        or extra[0].GetTotalNumHs(includeNeighbors=True) != 2
                        or mol.GetBondBetweenAtoms(d.GetIdx(), extra[0].GetIdx()).GetBondTypeAsDouble() != 1):
                    continue
                owned.add(extra[0].GetIdx())
                stem = 'carbonohydrazonoyl'
            else:
                continue
            if frozenset(owned) == heavy:
                rows.append(_decision(mol, stem + ' dicyanide', (cidx,), 'P-66.5.3.1/P-41'))
        elif len(nitriles) == 1 and d.GetAtomicNum() == 8 and d.GetDegree() == 1:
            other = [n for n in singles if n.GetIdx() != nitriles[0][0]]
            if len(other) != 1:
                continue
            ligand = other[0]
            if ligand.GetAtomicNum() == 7 and not ligand.IsInRing() and ligand.GetTotalValence() == 3:
                seen = {cidx, d.GetIdx(), ligand.GetIdx(), *nitriles[0][1]}
                roots = [n.GetIdx() for n in ligand.GetNeighbors() if n.GetIdx() != cidx]
                if len(roots) > 2 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in roots):
                    continue
                from .functional_replacement_acids import _organyl_units
                from .formatting import format_prefixes
                # Graph-own branches first; reproduce the amide N locants separately.
                if _organyl_units(mol, Chem, ligand.GetIdx(), roots, seen) is None or frozenset(seen) != heavy:
                    continue
                from .component_mapping import mapped_branch
                terms = [('N', mapped_branch(mol, Chem, root, ligand.GetIdx())[0]) for root in roots]
                prefix = format_prefixes(terms)
                rows.append(_decision(mol, prefix + 'carbamoyl cyanide', (cidx,), 'P-66.5.3.1/P-66.1'))
            elif ligand.GetAtomicNum() == 8 and ligand.GetDegree() == 2:
                other_c = [n for n in ligand.GetNeighbors() if n.GetIdx() != cidx]
                if len(other_c) != 1 or other_c[0].GetAtomicNum() != 6:
                    continue
                oc = other_c[0]
                ocidx = oc.GetIdx()
                oxo = [n for n in oc.GetNeighbors() if n.GetAtomicNum() == 8 and n.GetDegree() == 1
                       and mol.GetBondBetweenAtoms(ocidx, n.GetIdx()).GetBondTypeAsDouble() == 2]
                cyano = [_cyano(mol, n.GetIdx(), ocidx) for n in oc.GetNeighbors() if n.GetAtomicNum() == 6]
                cyano = [a for a in cyano if a]
                if len(oxo) == len(cyano) == 1:
                    atoms = {cidx, d.GetIdx(), ligand.GetIdx(), ocidx, oxo[0].GetIdx(),
                             *nitriles[0][1], *cyano[0]}
                    if frozenset(atoms) == heavy:
                        rows.append(_decision(mol, 'dicarbonic dicyanide', (cidx, ocidx), 'P-66.5.3.1/P-65.5.3.2'))
    # The two carbonyl centers of a symmetrical dicarbonic acid are equivalent.
    names = {r.name for r in rows}
    return rows[0] if len(names) == 1 else None
