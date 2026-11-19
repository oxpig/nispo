"""P-65.1.6.1/P-65.2.1.5 vocabulary for mapped ordinary carbamoyl units."""
from __future__ import annotations


def mapped_n_substituted_carbamoyl(mol, Chem, root, host, nitrogen, ligand=None):
    """Construct an ordinary carbamoyl unit on an existing C/N/O parent cut.

    The branch writer has already identified C=O and its nitrogen. Zero, one
    or two carbon-bound N ligands retain disjoint original-atom maps, locants
    and configuration. This supplies no parent-selection or whole-name
    preference evidence. Unsupported annotations/ligands keep their writers.
    The optional single-ligand argument preserves the earlier local contract.
    """
    carbon = mol.GetAtomWithIdx(root)
    amine = mol.GetAtomWithIdx(nitrogen)
    parent = mol.GetAtomWithIdx(host)
    cut = mol.GetBondBetweenAtoms(root, host)
    ligands = tuple(a.GetIdx() for a in amine.GetNeighbors() if a.GetIdx() != root)
    if (carbon.IsInRing() or carbon.GetDegree() != 3 or carbon.GetTotalValence() != 4
            or parent.GetAtomicNum() not in {6, 7, 8}
            or cut is None or cut.GetBondTypeAsDouble() != 1
            or amine.IsInRing() or amine.GetTotalValence() != 3
            or len(ligands) > 2
            or amine.GetTotalNumHs(includeNeighbors=True) != 2 - len(ligands)
            or ligand is not None and ligands != (ligand,)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetBondBetweenAtoms(nitrogen, i).GetBondTypeAsDouble() != 1
                   for i in ligands)):
        return None
    if parent.GetAtomicNum() in {7, 8} and (
            parent.GetFormalCharge() or parent.GetIsotope() or parent.GetNumRadicalElectrons()
            or int(parent.GetChiralTag()) or parent.GetIsAromatic()
            or parent.GetTotalValence() != (3 if parent.GetAtomicNum() == 7 else 2)
            or parent.GetDegree() not in ({2, 3} if parent.GetAtomicNum() == 7 else {2})
            or any(b.GetBondTypeAsDouble() != 1 for b in parent.GetBonds())):
        return None
    oxygen = [a.GetIdx() for a in carbon.GetNeighbors()
              if a.GetAtomicNum() == 8 and a.GetDegree() == 1
              and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondTypeAsDouble() == 2]
    if (len(oxygen) != 1
            or {a.GetIdx() for a in carbon.GetNeighbors()} != {host, nitrogen, oxygen[0]}
            or mol.GetBondBetweenAtoms(root, nitrogen).GetBondTypeAsDouble() != 1):
        return None
    core = frozenset({root, oxygen[0], nitrogen})
    if any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or int(a.GetChiralTag()) for a in (carbon, amine, mol.GetAtomWithIdx(oxygen[0]))):
        return None

    from .branch.functional_groups import collect_subtree
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms

    # The outer branch already collects this directed carbonyl cut. Its
    # non-ring C and N and their exact neighbor sets leave separate ligand
    # components. Reuse that scope; each mapped child independently proves
    # its own cut and the complete union below must account for every atom.
    scope = frozenset(collect_subtree(mol, root, host))
    owned = scope - core
    atoms = tuple(mol.GetAtomWithIdx(i) for i in owned)
    if (host in scope or not core <= scope
            or any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in atoms)):
        return None
    if any(any(a.GetIdx() in scope for a in group.GetAtoms()) for group in mol.GetStereoGroups()):
        return None
    rings = frozenset(a.GetIdx() for a in atoms if a.IsInRing())
    embedded, witnessed, units = set(), set(), []
    for child in ligands:
        name, mapped = mapped_branch(mol, Chem, child, nitrogen)
        mapped = frozenset(mapped)
        if (not name or child not in mapped or mapped & (core | {host} | embedded)
                or not mapped <= owned):
            return None
        embedded.update(mapped)
        witnessed.update(mapped_branch_ring_atoms(mol, Chem, child, nitrogen, name, mapped))
        units.append((name, child, mapped))
    if embedded != owned or not rings <= witnessed:
        return None
    prefix = _carbamoyl_ligand_sequence(mol, units)
    if prefix is None:
        return None
    text = prefix + 'carbamoyl'
    _record_carbamoyl_construction(mol, Chem, root, host, text, scope, rings)
    return text, scope


def _carbamoyl_ligand_sequence(mol, units):
    """P-14/P-16 composition of complete, separately mapped N ligands."""
    from .formatting import enclose_name_unit, prefix_alphabetization_key

    if not units:
        return ''
    if len(units) == 1:
        name = units[0][0]
        return (enclose_name_unit(name) if name.endswith('oxycarbonyl')
                or any(ch in name for ch in '-()[]{}') else name)
    from .organyl_prefix_components import organyl_prefix_nature, multiplied_organyl_prefix
    from .component_vocabulary import SUBSTITUENT_VOCABULARY
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence

    natures = []
    for name, child, mapped in units:
        nature = organyl_prefix_nature(mol, child, mapped, name)
        if nature is None and multiplied_organyl_prefix(mol, child, mapped, name, 1) == name:
            nature = 'compound'
        retained = SUBSTITUENT_VOCABULARY.get(name)
        if retained is not None and retained.source_status == 'preferred-retained':
            # Retention has already been proved on this complete mapped unit.
            nature = 'simple'
        natures.append(nature)
    sequence = mononuclear_prefix_sequence(
        PrefixUnit(unit[0], nature) for unit, nature in zip(units, natures))
    if sequence is not None:
        return sequence.text
    if units[0][0] == units[1][0]:
        name = units[0][0]
        if natures[0] is None or natures[0] != natures[1]:
            return None
        from .formatting import _multiplied_prefix_unit_name, format_prefixes
        from .substituent_multiplication import multiply_substituent
        plain = _multiplied_prefix_unit_name(name)
        operation = multiply_substituent(plain, 2, nature=natures[0])
        if operation is not None:
            # P-91 descriptors do not change the component's constitutional
            # simple/compound nature. Keep each identical descriptor inside
            # the multiplied complete component, never on the carbamoyl N.
            return (operation.multiplier + enclose_name_unit(name)
                    if plain != name else operation.text)
        retained = SUBSTITUENT_VOCABULARY.get(name)
        if (retained is not None and retained.source_status == 'preferred-retained'
                and name.isalpha() and natures[0] == 'simple'):
            return format_prefixes([('', name), ('', name)], omit_locants=True)
        return None
    ordered = sorted(zip(units, natures), key=lambda row: prefix_alphabetization_key(row[0][0]))
    first, second = (row[0][0] for row in ordered)
    # Stereochemical/alphanumerical ties need their separate comparison;
    # preserve the existing writer instead of selecting by atom storage order.
    if prefix_alphabetization_key(first)[:-1] == prefix_alphabetization_key(second)[:-1]:
        return None
    first = (enclose_name_unit(first) if ordered[0][1] == 'compound'
             or any(ch in first for ch in '-()[]{}') else first)
    return first + enclose_name_unit(second)


def _record_carbamoyl_construction(mol, Chem, root, host, text, scope, rings):
    from .component_mapping import _record_branch_ring_atoms
    _record_branch_ring_atoms(mol, Chem, root, host, text, scope, rings)
    # The enclosing cyclic-branch mapper can reuse this complete construction
    # instead of rebuilding an amino/oxo carbon component. Native state binds
    # the proof to annotations and temporary modes as well as connectivity.
    from .molecule_cache import mol_cache_dict
    try:
        state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    else:
        mol_cache_dict(mol, 'mapped_carbamoyl_prefixes')[id(Chem), root, host] = (state, text, scope)


def proved_carbamoyl_construction(mol, Chem, root, host, text, scope):
    """Verify the local producer's complete construction on the same graph."""
    from .molecule_cache import mol_cache_get
    cached = mol_cache_get(mol, 'mapped_carbamoyl_prefixes', {}).get((id(Chem), root, host))
    if cached is None or cached[1:] != (text, scope):
        return False
    return cached[0] == mol.ToBinary(Chem.PropertyPickleOptions.AllProps)


def mapped_carbamoyl_linker(mol, Chem, root, host, names):
    """Join a proved carbamoyl to its original ordinary amino/oxy linker.

    Only already constructed acyl units qualify. This preserves the chosen
    parent cut and carries the child's ring ownership onto the whole prefix.
    One additional carbon ligand on an amino linker is mapped independently.
    """
    from .molecule_cache import mol_cache_get
    proofs = mol_cache_get(mol, 'mapped_carbamoyl_prefixes', {})
    if not proofs:
        return None
    atom = mol.GetAtomWithIdx(root)
    children = tuple(a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() != host)
    acyl = [(i, proofs.get((id(Chem), i, root))) for i in children
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
    acyl = [(i, p) for i, p in acyl if p is not None]
    bond = mol.GetBondBetweenAtoms(root, host)
    if (len(acyl) != 1 or len(children) != len(names)
            or atom.GetAtomicNum() not in {7, 8} or atom.IsInRing()
            or atom.GetIsotope() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
            or int(atom.GetChiralTag()) or mol.GetAtomWithIdx(host).GetAtomicNum() != 6
            or bond is None or bond.GetBondTypeAsDouble() != 1
            or atom.GetTotalValence() != (3 if atom.GetAtomicNum() == 7 else 2)
            or len(children) not in ({1, 2} if atom.GetAtomicNum() == 7 else {1})
            or atom.GetTotalNumHs(includeNeighbors=True) != (2 - len(children) if atom.GetAtomicNum() == 7 else 0)
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
        return None
    child, proof = acyl[0]
    try:
        state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if proof[0] != state or proof[1] != names[children.index(child)]:
        return None
    from .branch.functional_groups import collect_subtree
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms
    from .formatting import enclose_name_unit, prefix_alphabetization_key

    embedded = set(proof[2])
    if embedded & {root, host}:
        return None
    rings = set(mapped_branch_ring_atoms(mol, Chem, child, root, proof[1], proof[2]))
    distal = [a for a in mol.GetAtomWithIdx(child).GetNeighbors()
              if a.GetIdx() != root and a.GetAtomicNum() == 7]
    if len(distal) != 1:
        return None
    simple = distal[0].GetTotalNumHs(includeNeighbors=True) == 2
    terms = [(proof[1], 'simple' if simple else 'compound')]
    for other in children:
        if other == child:
            continue
        if mol.GetAtomWithIdx(other).GetAtomicNum() != 6:
            return None
        text, owned = mapped_branch(mol, Chem, other, root)
        if not text or set(owned) & (embedded | {root, host}):
            return None
        if any(mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
               or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in owned):
            return None
        embedded.update(owned)
        rings.update(mapped_branch_ring_atoms(mol, Chem, other, root, text, owned))
        from .organyl_prefix_components import organyl_prefix_nature
        nature = organyl_prefix_nature(mol, other, owned, text)
        terms.append((text, nature))
    scope = frozenset(embedded | {root})
    if scope != frozenset(collect_subtree(mol, root, host)):
        return None
    if any(any(a.GetIdx() in scope for a in g.GetAtoms()) for g in mol.GetStereoGroups()):
        return None
    if len(terms) == 1:
        prefix = terms[0][0] if simple else enclose_name_unit(terms[0][0])
    else:
        terms.sort(key=lambda unit: prefix_alphabetization_key(unit[0]))
        if prefix_alphabetization_key(terms[0][0])[:-1] == prefix_alphabetization_key(terms[1][0])[:-1]:
            return None
        first, nature = terms[0]
        prefix = (enclose_name_unit(first) if nature == 'compound'
                  or any(ch in first for ch in '-()[]{}') else first) + enclose_name_unit(terms[1][0])
    text = prefix + ('amino' if atom.GetAtomicNum() == 7 else 'oxy')
    _record_carbamoyl_construction(mol, Chem, root, host, text, scope, rings)
    return text, scope


def mapped_carbamoylmethoxy(mol, Chem, root, host, scope):
    """Bind a carbamoyl group through a source O–CH2 cut."""
    oxygen = mol.GetAtomWithIdx(root)
    if (oxygen.GetDegree() != 2 or oxygen.IsInRing() or oxygen.GetFormalCharge()
            or oxygen.GetIsotope() or oxygen.GetNumRadicalElectrons()
            or any(b.GetBondTypeAsDouble() != 1 for b in oxygen.GetBonds())):
        return None
    methylene = next(a for a in oxygen.GetNeighbors() if a.GetIdx() != host)
    if (methylene.GetAtomicNum() != 6 or methylene.GetDegree() != 2 or methylene.IsInRing()
            or methylene.GetTotalNumHs() != 2 or methylene.GetIsotope()
            or methylene.GetFormalCharge()):
        return None
    linker = methylene.GetIdx()
    acyl = next(a for a in methylene.GetNeighbors() if a.GetIdx() != root)
    if acyl.GetAtomicNum() != 6 or acyl.IsInRing() or acyl.GetDegree() != 3:
        return None
    nitrogen = [a.GetIdx() for a in acyl.GetNeighbors() if a.GetAtomicNum() == 7]
    if len(nitrogen) != 1:
        return None
    child = acyl.GetIdx()
    mapped = mapped_n_substituted_carbamoyl(mol, Chem, child, linker, nitrogen[0])
    if mapped is None or mapped[1] != scope - {root, linker}:
        return None
    from .component_mapping import mapped_branch_ring_atoms
    from .formatting import enclose_name_unit

    carbamoyl, atoms = mapped
    prefix = carbamoyl if carbamoyl == 'carbamoyl' else enclose_name_unit(carbamoyl)
    text = prefix + 'methoxy'
    rings = mapped_branch_ring_atoms(mol, Chem, child, linker, carbamoyl, atoms)
    _record_carbamoyl_construction(mol, Chem, root, host, text, scope, rings)
    return text, scope


def preferred_carbamoyl_prefix(mol, root, host, emitted, owned):
    """Qualify a legacy branch token with its exact original neutral graph.

    Only an unsubstituted or plain N-methyl/N,N-dimethyl unit is admitted.
    Labels, unusual valence and larger ligands keep their existing writers.
    This is a local prefix operation, never a rewrite of a complete name.
    """
    vocabulary = {"carbamyl": (0, "carbamoyl"),
                  "methylcarbamyl": (1, "methylcarbamoyl"),
                  "dimethylcarbamyl": (2, "dimethylcarbamoyl")}
    vocabulary.update({text: (count, text) for count, text in tuple(vocabulary.values())})
    entry = vocabulary.get(emitted)
    if entry is None:
        return None
    carbon = mol.GetAtomWithIdx(root)
    bond = mol.GetBondBetweenAtoms(root, host)
    if (carbon.GetAtomicNum() != 6 or carbon.IsInRing()
            or bond is None or bond.GetBondTypeAsDouble() != 1
            or carbon.GetTotalValence() != 4):
        return None
    oxygen, nitrogen = [], []
    for atom in carbon.GetNeighbors():
        if atom.GetIdx() == host:
            continue
        order = mol.GetBondBetweenAtoms(root, atom.GetIdx()).GetBondTypeAsDouble()
        if atom.GetAtomicNum() == 8 and order == 2:
            oxygen.append(atom)
        elif atom.GetAtomicNum() == 7 and order == 1:
            nitrogen.append(atom)
        else:
            return None
    if len(oxygen) != 1 or len(nitrogen) != 1:
        return None
    nitrogen = nitrogen[0]
    ligands = [atom for atom in nitrogen.GetNeighbors()
               if atom.GetIdx() != root and atom.GetAtomicNum() > 1]
    if (len(ligands) != entry[0] or nitrogen.IsInRing()
            or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 2 - len(ligands)
            or oxygen[0].GetDegree() != 1):
        return None
    for atom in ligands:
        if (atom.GetAtomicNum() != 6 or atom.GetDegree() != 1
                or atom.GetTotalNumHs(includeNeighbors=True) != 3
                or mol.GetBondBetweenAtoms(nitrogen.GetIdx(), atom.GetIdx()).GetBondTypeAsDouble() != 1):
            return None
    scope = {root, oxygen[0].GetIdx(), nitrogen.GetIdx(), *(atom.GetIdx() for atom in ligands)}
    if scope != set(owned):
        return None
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge()
           or mol.GetAtomWithIdx(idx).GetIsotope()
           or mol.GetAtomWithIdx(idx).GetNumRadicalElectrons()
           or int(mol.GetAtomWithIdx(idx).GetChiralTag())
           or any(atom.GetAtomicNum() == 1 and atom.GetIsotope()
                  for atom in mol.GetAtomWithIdx(idx).GetNeighbors())
           for idx in scope):
        return None
    return entry[1]
