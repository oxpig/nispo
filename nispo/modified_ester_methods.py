"""Mapped, bounded esters of functionally modified and sulfuric acids."""

from .naming_methods import MethodDecision

RULE = "P-65.1.3/P-65.1.5/P-65.6.3.3.7.1"


def _plain_graph(mol, Chem):
    return (not mol.GetStereoGroups() and len(Chem.GetMolFrags(mol)) == 1
            and all(not (a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                         or int(a.GetChiralTag()) or a.HasQuery() or a.GetAtomicNum() == 1)
                    for a in mol.GetAtoms())
            and all(not (int(b.GetStereo()) or b.HasQuery() or int(b.GetBondDir()))
                    for b in mol.GetBonds()))


def _systematic_acid_stem(acid_name):
    """Return the parent stem and whether the acid C is an exocyclic unit."""
    retained = {"formic acid": "methan", "acetic acid": "ethan"}
    if acid_name in retained:
        return retained[acid_name], False
    if acid_name.endswith("benzoic acid"):
        # Substitution does not move the acid carbon into the benzene ring.
        # Preserve that exocyclic role for every substituted retained parent,
        # rather than treating e.g. methylbenz- as an alkane suffix stem.
        return acid_name[:-len("benzoic acid")] + "benzene", True
    if acid_name.endswith("carboxylic acid"):
        return acid_name[:-len("carboxylic acid")], True
    if acid_name.endswith("oic acid"):
        return acid_name[:-len("oic acid")], False
    return None


def _modified_anion(stem, exocyclic, double_kind, bridge_number):
    elements = {16: "thio", 34: "seleno", 52: "telluro"}
    replacements = [elements[n] for n in (double_kind, bridge_number) if n in elements]
    if double_kind == 7:
        replacement = ("imido" + "".join(sorted(replacements))) if replacements else "imid"
        suffix = "carbox" + replacement + "ate" if exocyclic else replacement + "ate"
    elif double_kind == "hydrazono":
        replacement = ("hydrazono" + "".join(sorted(replacements))) if replacements else "hydrazon"
        suffix = "carbo" + replacement + "ate" if exocyclic else "e" + replacement + "ate"
    else:
        if not replacements:
            return ""
        replacement = ("di" + replacements[0] if len(replacements) == 2
                       and replacements[0] == replacements[1] else "".join(sorted(replacements)))
        ending = replacement + "ate"
        if exocyclic:
            suffix = "carbo" + ending
        elif replacement.startswith("di"):
            suffix = "e(" + ending + ")"
        else:
            suffix = "e" + ending
    return stem + suffix


def modified_carboxylic_ester_decision(mol, Chem):
    """Prove one neutral carbon-only acid and organyl part with no senior class.

    The input is split at the chalcogen bridge, then the ordinary acid graph
    is restored solely to reuse its mapped parent selection. Only unlabelled
    components and terminal =NH/=NNH2 or chalcogen replacement are admitted.
    """
    if mol.GetNumHeavyAtoms() > 64 or not _plain_graph(mol, Chem):
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .component_mapping import mapped_branch
    from .acid_derivative_methods import _acid_component

    choices = []
    for carbon in mol.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetIsAromatic():
            continue
        center = carbon.GetIdx()
        double = [n for n in carbon.GetNeighbors()
                  if n.GetAtomicNum() in {7, 8, 16, 34, 52}
                  and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2]
        bridges = [n for n in carbon.GetNeighbors()
                   if n.GetAtomicNum() in {8, 16, 34, 52}
                   and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1
                   and n.GetDegree() == 2 and not n.IsInRing()]
        if len(double) != 1 or len(bridges) != 1:
            continue
        terminal, bridge = double[0], bridges[0]
        roots = [n for n in bridge.GetNeighbors() if n.GetIdx() != center]
        if len(roots) != 1 or roots[0].GetAtomicNum() != 6:
            continue
        root = roots[0].GetIdx()
        organic = _carbon_component(mol, root, bridge.GetIdx())
        if organic is None:
            continue
        acid_roots = [n for n in carbon.GetNeighbors()
                      if n.GetIdx() not in {terminal.GetIdx(), bridge.GetIdx()}]
        if len(acid_roots) > 1 or any(n.GetAtomicNum() != 6 for n in acid_roots):
            continue
        acid_carbon = (_carbon_component(mol, acid_roots[0].GetIdx(), center)
                       if acid_roots else frozenset())
        if acid_carbon is None:
            continue
        kind = terminal.GetAtomicNum()
        extra = set()
        if kind == 7:
            neighbors = [n for n in terminal.GetNeighbors() if n.GetIdx() != center]
            if not neighbors and terminal.GetTotalNumHs(includeNeighbors=True) == 1:
                pass
            elif (len(neighbors) == 1 and neighbors[0].GetAtomicNum() == 7
                  and neighbors[0].GetDegree() == 1
                  and neighbors[0].GetTotalNumHs(includeNeighbors=True) == 2
                  and mol.GetBondBetweenAtoms(terminal.GetIdx(), neighbors[0].GetIdx()).GetBondTypeAsDouble() == 1):
                extra.add(neighbors[0].GetIdx())
                kind = "hydrazono"
            else:
                continue
        elif terminal.GetDegree() != 1:
            continue
        bridge_num = bridge.GetAtomicNum()
        if kind == 8 and bridge_num == 8:
            continue
        acid = set(acid_carbon) | {center, terminal.GetIdx()} | extra
        organyl = set(organic) | {bridge.GetIdx()}
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        if acid & organyl or acid | organyl != heavy:
            continue
        # The normalized temporary molecule retains original indices except
        # for an optional terminal hydrazino N, which is removed from the copy.
        copy = Chem.RWMol(mol)
        copy.GetAtomWithIdx(terminal.GetIdx()).SetAtomicNum(8)
        copy.GetAtomWithIdx(terminal.GetIdx()).SetNoImplicit(True)
        copy.GetAtomWithIdx(terminal.GetIdx()).SetNumExplicitHs(0)
        copy.GetAtomWithIdx(bridge.GetIdx()).SetAtomicNum(8)
        for idx in sorted(extra, reverse=True):
            copy.RemoveAtom(idx)
        def remap(idx):
            return idx - sum(i < idx for i in extra)
        normalized = copy.GetMol()
        try:
            Chem.SanitizeMol(normalized)
        except (ValueError, RuntimeError):
            continue
        part = _acid_component(normalized, Chem, remap(center), remap(bridge.GetIdx()))
        stem = _systematic_acid_stem(part.name) if part is not None else None
        word, mapped = mapped_branch(mol, Chem, root, bridge.GetIdx())
        if stem is None or not word or frozenset(mapped) != organic:
            continue
        anion = _modified_anion(*stem, kind, bridge_num)
        if not anion:
            continue
        # Element symbols distinguish C=O/C=S etc. tautomers when needed.
        if kind not in {7, "hydrazono"} and kind != bridge_num:
            word = bridge.GetSymbol() + "-" + ("(" + word + ")" if "-" in word else word)
        roles = (("modified-acid", frozenset(acid), ()),
                 ("organyl-and-link", frozenset(organyl), ()))
        choices.append(MethodDecision(word + " " + anion, "functional-class",
                                      frozenset({center}), frozenset(heavy), RULE, (),
                                      component_roles=roles))
    return choices[0] if len(choices) == 1 else None


def sulfuric_ester_decision(mol, Chem):
    """P-67.1.3.2 for neutral sulfate mono- and diesters of carbon groups."""
    if mol.GetNumHeavyAtoms() > 64 or not _plain_graph(mol, Chem):
        return None
    if any(a.GetAtomicNum() not in {6, 8, 16} for a in mol.GetAtoms()):
        return None
    sulfurs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 16]
    if len(sulfurs) != 1:
        return None
    sulfur = sulfurs[0]
    if sulfur.IsInRing() or len(sulfur.GetNeighbors()) != 4:
        return None
    oxo, hydroxyl, esters = [], [], []
    from .phosphorus_ester_methods import _carbon_component, _organyl_words
    for oxygen in sulfur.GetNeighbors():
        if oxygen.GetAtomicNum() != 8 or oxygen.IsInRing():
            return None
        order = mol.GetBondBetweenAtoms(sulfur.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble()
        if order == 2 and oxygen.GetDegree() == 1:
            oxo.append(oxygen.GetIdx())
        elif order == 1:
            others = [n for n in oxygen.GetNeighbors() if n.GetIdx() != sulfur.GetIdx()]
            if not others and oxygen.GetTotalNumHs(includeNeighbors=True) == 1:
                hydroxyl.append(oxygen.GetIdx())
            elif len(others) == 1 and others[0].GetAtomicNum() == 6:
                root = others[0].GetIdx()
                component = _carbon_component(mol, root, oxygen.GetIdx())
                if component is None:
                    return None
                esters.append((oxygen.GetIdx(), root, component))
            else:
                return None
        else:
            return None
    if len(oxo) != 2 or len(esters) not in {1, 2} or len(hydroxyl) + len(esters) != 2:
        return None
    words = _organyl_words(mol, Chem, {"ester_oxygens": esters})
    if words is None:
        return None
    core = frozenset({sulfur.GetIdx(), *oxo, *hydroxyl})
    roles = [("sulfate-anion", core, ())]
    for i, (_key, _word, rows) in enumerate(words, 1):
        roles.append((f"organyl:{i}", frozenset().union(*(frozenset(r["owned"]) for r in rows)), ()))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if sum(len(row[1]) for row in roles) != len(heavy) or frozenset().union(*(r[1] for r in roles)) != heavy:
        return None
    name = " ".join([*(row[1] for row in words), *(["hydrogen"] if hydroxyl else []), "sulfate"])
    return MethodDecision(name, "functional-class", frozenset({sulfur.GetIdx()}), heavy,
                          "P-67.1.3.2", (), component_roles=tuple(roles))


def modified_sulfur_ester_decision(mol, Chem):
    """Name simple chalcogen analogues of sulfonate and sulfinate esters.

    A private O-normalized copy is passed to the existing mapped sulfur-acid
    parent selector; the returned ownership remains on the original graph.
    """
    if mol.GetNumHeavyAtoms() > 64 or not _plain_graph(mol, Chem):
        return None
    if any(a.GetAtomicNum() not in {6, 8, 16, 34, 52} for a in mol.GetAtoms()):
        return None
    from .phosphorus_ester_methods import _carbon_component
    from .sulfur_ester_methods import sulfur_ester_frontier

    choices = []
    for center in mol.GetAtoms():
        if center.GetAtomicNum() != 16 or center.IsInRing():
            continue
        idx = center.GetIdx()
        double = [n for n in center.GetNeighbors()
                  if n.GetAtomicNum() in {8, 16, 34, 52} and n.GetDegree() == 1
                  and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 2]
        links = [n for n in center.GetNeighbors()
                 if n.GetAtomicNum() in {8, 16, 34, 52} and n.GetDegree() == 2
                 and not n.IsInRing()
                 and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        roots = [n for n in center.GetNeighbors() if n.GetAtomicNum() == 6
                 and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1]
        if (len(double) not in {1, 2} or len(links) != 1 or len(roots) != 1
                or center.GetDegree() != len(double) + 2):
            continue
        link = links[0]
        other = [n for n in link.GetNeighbors() if n.GetIdx() != idx]
        if len(other) != 1 or other[0].GetAtomicNum() != 6:
            continue
        acid_c = _carbon_component(mol, roots[0].GetIdx(), idx)
        organyl_c = _carbon_component(mol, other[0].GetIdx(), link.GetIdx())
        if acid_c is None or organyl_c is None or acid_c & organyl_c:
            continue
        acid_atoms = set(acid_c) | {idx, *(n.GetIdx() for n in double)}
        organyl_atoms = set(organyl_c) | {link.GetIdx()}
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if acid_atoms | organyl_atoms != set(heavy):
            continue
        replacements = [n.GetAtomicNum() for n in (*double, link) if n.GetAtomicNum() != 8]
        if not replacements:
            continue
        copy = Chem.RWMol(mol)
        for atom in (*double, link):
            copy.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(8)
        normalized = copy.GetMol()
        try:
            Chem.SanitizeMol(normalized)
        except (ValueError, RuntimeError):
            continue
        frontier = sulfur_ester_frontier(normalized, Chem)
        if frontier.disposition != 'complete' or frontier.decision is None:
            continue
        word, sep, anion = frontier.decision.name.rpartition(' ')
        ending = 'sulfonate' if len(double) == 2 else 'sulfinate'
        if not sep or not anion.endswith(ending):
            continue
        affixes = [{16: 'thio', 34: 'seleno', 52: 'telluro'}[n] for n in replacements]
        if len(affixes) == 2 and affixes[0] == affixes[1]:
            infix = 'di' + affixes[0]
        elif len(affixes) == 3 and len(set(affixes)) == 1:
            infix = 'tri' + affixes[0]
        else:
            infix = ''.join(sorted(affixes))
        base = 'sulfono' if len(double) == 2 else 'sulfino'
        anion = anion[:-len(ending)] + base + infix + 'ate'
        if len({n.GetAtomicNum() for n in (*double, link)}) > 1:
            word = link.GetSymbol() + '-' + ('(' + word + ')' if '-' in word else word)
        roles = (("modified-sulfur-acid", frozenset(acid_atoms), ()),
                 ("organyl-and-link", frozenset(organyl_atoms), ()))
        choices.append(MethodDecision(word + ' ' + anion, 'functional-class',
                                      frozenset({idx}), heavy, RULE, (),
                                      component_roles=roles))
    return choices[0] if len(choices) == 1 else None
