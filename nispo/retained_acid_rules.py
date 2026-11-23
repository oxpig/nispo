"""Source-bounded facts for locant omission on the retained acetic parent.

P-14.3.4.6 permits omission when all substitutable hydrogen atoms have one
locant.  This module proves that condition from the mapped two-carbon acid
component; it does not infer it from the rendered parent text alone.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class RetainedAcidLocantEvidence:
    parent_class: str
    principal_kind: str
    carbonyl_atom: int
    alpha_carbon: int
    parent_atoms: frozenset[int]
    alpha_locant: str
    prefix_texts: tuple[str, ...]
    rule: str


def retained_acid_locant_evidence(
    mol, Chem, *, parent_class, principal_kind, parent_atoms,
    atom_locants, prefixes, isotope_forces_locants=False,
):
    """Return evidence only for source-closed acetic-acid locant omission."""
    if (parent_class != "acetic acid" or principal_kind not in
            {"acid", "carboxylate", "ester"} or isotope_forces_locants):
        return None
    parent_atoms = frozenset(parent_atoms)
    locants = dict(atom_locants)
    if not parent_atoms or any(i not in locants for i in parent_atoms
                               if mol.GetAtomWithIdx(i).GetAtomicNum() > 1):
        return None
    carbons = {i for i in parent_atoms
               if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    if len(carbons) != 2:
        return None
    carbonyls = []
    for i in carbons:
        atom = mol.GetAtomWithIdx(i)
        double_oxygens = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 8
                          and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondTypeAsDouble() == 2]
        if len(double_oxygens) == 1:
            carbonyls.append((i, double_oxygens[0]))
    if len(carbonyls) != 1:
        return None
    carbonyl, carbonyl_oxygen = carbonyls[0]
    if carbonyl_oxygen not in parent_atoms:
        return None
    alpha_candidates = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                        if n.GetIdx() in carbons and n.GetIdx() != carbonyl]
    if len(alpha_candidates) != 1:
        return None
    alpha = alpha_candidates[0]
    single_oxygens = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                      if n.GetAtomicNum() == 8
                      and mol.GetBondBetweenAtoms(carbonyl, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(single_oxygens) != 1:
        return None
    non_oxygen_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                            if n.GetAtomicNum() != 8]
    if non_oxygen_neighbors != [alpha]:
        return None
    # The second carbon is the only carbon bearing substitutable hydrogen in
    # this retained parent; P-14.3.4.5 excludes the acid hydroxyl hydrogen.
    if any(i not in {carbonyl, alpha} for i in carbons):
        return None
    alpha_locant = locants.get(alpha)
    if not isinstance(alpha_locant, str) or not alpha_locant:
        return None
    prefix_rows = tuple(prefixes)
    if not prefix_rows or any(loc != alpha_locant for loc, _ in prefix_rows):
        return None
    if any(not isinstance(text, str) or not text for _, text in prefix_rows):
        return None
    rule = (
        "P-14.3.4.5; P-14.3.4.6; P-65.6.3.3.1"
        if principal_kind == "ester"
        else "P-14.3.4.5; P-14.3.4.6; P-65.1.1.1"
    )
    return RetainedAcidLocantEvidence(
        "acetic acid", principal_kind, carbonyl, alpha, parent_atoms,
        alpha_locant, tuple(text for _, text in prefix_rows), rule,
    )


def validate_retained_acid_locant_evidence(mol, Chem, evidence, atom_locants, prefixes):
    """Recheck the typed graph proof at the final component-writing boundary."""
    if not isinstance(evidence, RetainedAcidLocantEvidence):
        return False
    if (evidence.parent_class != "acetic acid"
            or evidence.principal_kind not in {"acid", "carboxylate", "ester"}
            or evidence.rule not in {
                "P-14.3.4.5; P-14.3.4.6; P-65.1.1.1",
                "P-14.3.4.5; P-14.3.4.6; P-65.6.3.3.1",
            }
            or not isinstance(evidence.parent_atoms, frozenset)
            or len(evidence.parent_atoms) == 0
            or any(i < 0 or i >= mol.GetNumAtoms() for i in evidence.parent_atoms)
            or evidence.carbonyl_atom not in evidence.parent_atoms
            or evidence.alpha_carbon not in evidence.parent_atoms):
        return False
    recomputed = retained_acid_locant_evidence(
        mol, Chem, parent_class=evidence.parent_class,
        principal_kind=evidence.principal_kind,
        parent_atoms=evidence.parent_atoms,
        atom_locants=atom_locants, prefixes=prefixes,
        isotope_forces_locants=any(
            mol.GetAtomWithIdx(i).GetIsotope() for i in evidence.parent_atoms
        ),
    )
    return recomputed == evidence


def retained_acid_prefix_sequence(mol, Chem, evidence, atom_locants, prefixes):
    """Build the P-16 sequence from original mapped alpha-carbon branches.

    The caller must already hold the P-14.3.4.6 retained-parent evidence.
    Branch text is checked against a fresh mapped construction and the
    component nature comes from original graph roles, never from the spelling.
    """
    if not validate_retained_acid_locant_evidence(
            mol, Chem, evidence, atom_locants, prefixes):
        return None
    rows = tuple(prefixes)
    if not rows:
        return None
    alpha = mol.GetAtomWithIdx(evidence.alpha_carbon)
    branch_roots = [neighbor.GetIdx() for neighbor in alpha.GetNeighbors()
                    if neighbor.GetIdx() != evidence.carbonyl_atom
                    and neighbor.GetAtomicNum() > 1]
    if len(branch_roots) != len(rows):
        return None

    from .component_mapping import mapped_branch
    from .prefix_sequences import mononuclear_prefix_sequence
    from .organyl_prefix_components import organyl_prefix_nature
    from .extended_inventory import selection_inventory

    typed = []
    source_rules = {"P-14.3.4.5", "P-14.3.4.6", "P-16.5.1.3.2"}
    used_roots = set()
    used_atoms = set()
    inventory = selection_inventory(mol)
    for locant, text in rows:
        if locant != evidence.alpha_locant:
            return None
        matches = []
        for root in branch_roots:
            if root in used_roots:
                continue
            rendered, atoms = mapped_branch(mol, Chem, root, evidence.alpha_carbon)
            if rendered == text and atoms:
                matches.append((root, atoms))
        if not matches:
            return None
        # Equal component names can come from several structurally separate
        # branches (for example repeated bromo groups). Consume one original
        # branch per source row; the completed sequence coalesces the equal
        # component identities for multiplication.
        root, atoms = min(matches, key=lambda row: row[0])
        proof = _retained_prefix_nature(
            mol, Chem, root, atoms, text, evidence.alpha_carbon,
            inventory, organyl_prefix_nature,
        )
        if proof is None:
            return None
        nature, rules, component_name = proof
        source_rules.update(rules)
        if used_atoms & set(atoms):
            return None
        used_roots.add(root)
        used_atoms.update(atoms)
        typed.append((component_name or text, nature))
    if used_roots != set(branch_roots):
        return None
    if (len(typed) == 1 and typed[0][1] == 'compound'
            and any(mark in typed[0][0] for mark in '([]{}')):
        # The ordinary sequence API deliberately declines nested component
        # spellings. Here there is no ordering or multiplication decision:
        # the exact single component has already been rebuilt and classified
        # against its complete original rooted graph above.
        from .formatting import enclose_name_unit, prefix_alphabetization_key
        from .prefix_sequences import OrderedPrefixUnit, PrefixSequence, RULES

        name = typed[0][0]
        enclosed = enclose_name_unit(name)
        return PrefixSequence(enclosed, (OrderedPrefixUnit(
            name, 'compound', 1, enclosed, prefix_alphabetization_key(name), True,
        ),), RULES + tuple(sorted(source_rules)))
    sequence = mononuclear_prefix_sequence(typed)
    if sequence is None:
        return None
    return replace(sequence, rules=sequence.rules + (
        *tuple(sorted(source_rules)),
    ))


def mapped_ring_stereo_prefix_proved(mol, Chem, root, host, atoms, text):
    """Rejoin an emitted ring prefix to its exact original stereo witnesses."""
    atoms = frozenset(atoms)
    if (root not in atoms or host in atoms or not mol.GetAtomWithIdx(root).IsInRing()
            or any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in atoms)):
        return False
    tagged = {idx for idx in atoms if int(mol.GetAtomWithIdx(idx).GetChiralTag())}
    if not tagged or any(int(bond.GetStereo()) or int(bond.GetBondDir())
                         for idx in atoms for bond in mol.GetAtomWithIdx(idx).GetBonds()):
        return False
    boundary = {(idx, neighbor.GetIdx()) for idx in atoms
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in atoms}
    if boundary != {(root, host)}:
        return False
    from .mapped_ring_parents import records, rooted_prefix
    candidates = records(mol, Chem, scope=atoms, attachment=root)
    if len(candidates) != 1:
        from .natural_products import mapped_steroid_yl
        from .spiro_component_rules import mapped_component_spiro_yl
        return (mapped_steroid_yl(mol, Chem, root, host, atoms) == (text, atoms)
                or mapped_component_spiro_yl(mol, Chem, root, host, atoms) == (text, atoms))
    if rooted_prefix(mol, Chem, root, host, atoms) != (text, atoms):
        return False
    record = candidates[0]
    construction = record.construction
    if (record.name != text or record.preference_issues or not record.annotation_coverage
            or construction is None or construction.atoms & construction.embedded_atoms
            or construction.atoms | construction.embedded_atoms != atoms):
        return False
    from .component_mapping import decorate_component, _mapped_stereochemistry_plan, mapped_branch
    from .stereochemistry import assigned_atom_stereochemistry
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    expected = {idx: assigned.get(idx) for idx in tagged & construction.atoms}
    if any(value not in {'R', 'S'} for value in expected.values()):
        return False
    plan = _mapped_stereochemistry_plan(mol, Chem, construction, dict(construction.atom_locants))
    facts = {} if plan is None else {idx: descriptor for idx, _locant, descriptor in plan[1]}
    if plan is None or facts != expected or plan[2] or decorate_component(mol, Chem, construction) != text:
        return False
    unclaimed = tagged - construction.atoms
    for child, site, owned, prefix in construction.prefix_branch_ownership:
        if not unclaimed & owned:
            continue
        if mapped_branch(mol, Chem, child, site) != (prefix, owned):
            return False
        linker = mol.GetAtomWithIdx(child)
        if not linker.IsInRing():
            # One neutral methylene may carry the independently mapped
            # stereochemical ring. Its cut and single child fix ownership.
            downstream = [a.GetIdx() for a in linker.GetNeighbors() if a.GetIdx() in owned]
            if (linker.GetAtomicNum() != 6 or linker.GetTotalNumHs() != 2
                    or linker.GetFormalCharge() or len(downstream) != 1
                    or any(b.GetBondTypeAsDouble() != 1 for b in linker.GetBonds())):
                return False
            child, site = downstream[0], child
            prefix, nested = mapped_branch(mol, Chem, child, site)
            if nested != owned - {site}:
                return False
        else:
            nested = owned
        if not mapped_ring_stereo_prefix_proved(mol, Chem, child, site, nested, prefix):
            return False
        unclaimed -= nested
    return not unclaimed


def _retained_prefix_nature(mol, Chem, root, atoms, text, host, inventory,
                            organyl_prefix_nature):
    """Classify one mapped one-position prefix from its original rooted graph.

    The rendered text is only checked against `mapped_branch`; element roles,
    bond topology, and source-class records determine the prefix family.
    """
    atoms = frozenset(atoms)
    if (atoms and any(int(mol.GetAtomWithIdx(idx).GetChiralTag()) for idx in atoms)
            and mapped_ring_stereo_prefix_proved(mol, Chem, root, host, atoms, text)):
        return 'compound', ('P-29', 'P-91', 'P-16.5.1.3.2'), text
    if atoms and root in atoms and mol.GetAtomWithIdx(root).GetAtomicNum() == 7:
        from .branch.amido_components import mapped_amido_prefix
        if mapped_amido_prefix(mol, Chem, root, host, atoms, chain_only=True) == (text, atoms):
            return 'compound', ('P-66.1.1.4.3', 'P-16.5.1.3.2'), text
    if (not atoms or root not in atoms or any(
            mol.GetAtomWithIdx(i).GetIsotope()
            or int(mol.GetAtomWithIdx(i).GetChiralTag())
            for i in atoms)):
        return None
    # A stereobond or isotope-labelled explicit H is not silently discarded
    # when the component mapper returns heavy-atom ownership only.
    for bond in mol.GetBonds():
        if ({bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & atoms
                and (int(bond.GetStereo()) or int(bond.GetBondDir()))):
            return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if any(n.GetAtomicNum() == 1 and n.GetIsotope()
               for n in atom.GetNeighbors()):
            return None

    atom = mol.GetAtomWithIdx(root)
    element = atom.GetAtomicNum()
    if element == 6:
        nitriles = [group for group in inventory.groups
                    if group.kind == 'nitrile' and group.center == root
                    and group.owner == host and frozenset(group.atoms) == atoms]
        if inventory.supported and len(nitriles) == 1 and text == 'cyano' and len(atoms) == 2:
            nitrogen = next((mol.GetAtomWithIdx(i) for i in atoms
                             if mol.GetAtomWithIdx(i).GetAtomicNum() == 7), None)
            attachment = mol.GetBondBetweenAtoms(root, host)
            triple = (mol.GetBondBetweenAtoms(root, nitrogen.GetIdx())
                      if nitrogen is not None else None)
            if (not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons()
                    and nitrogen is not None and not nitrogen.GetFormalCharge()
                    and not nitrogen.GetNumRadicalElectrons() and nitrogen.GetDegree() == 1
                    and attachment is not None and attachment.GetBondTypeAsDouble() == 1
                    and triple is not None and triple.GetBondTypeAsDouble() == 3):
                return 'simple', ('P-66.5.1.1.4', 'P-16.5.1.3.2'), text
        # A characteristic-group carbon is not an ordinary organyl path
        # extension. Bind its complete mapped core and attachment to the
        # recognized original group before using its prescribed prefix.
        characteristic = [group for group in inventory.groups
                          if group.kind in {"acid", "carboxylate"}
                          and group.center == root and group.owner == host
                          and frozenset(group.atoms) == atoms]
        if inventory.supported and len(characteristic) == 1:
            from .simple_prefixes import _terminal_carboxyl_atoms

            typed = _terminal_carboxyl_atoms(mol, root, host)
            expected = {"acid": "carboxy", "carboxylate": "carboxylato"}[characteristic[0].kind]
            if typed is not None and typed[0] == text == expected and frozenset(typed[1]) == atoms:
                return "simple", ("P-65.6.2.3.1", "P-16.5.1.3.2"), text
        # Preserve the source warning that an alkyl substituent on the
        # retained acetic-acid parent may change the selected parent.
        nature = organyl_prefix_nature(mol, root, atoms, text)
        if text == "phenyl" and nature == "simple":
            return "simple", ("P-29.3.4", "P-16.5.1.3.2"), text
        # A mapped aromatic ring branch has no extending acyclic carbon path.
        # Restrict this proof to ring atoms plus terminal ring-bound halogens,
        # methyl groups, or source-recognized ring-lactam oxygens, with every
        # ring atom witnessed by the actual mapped ring writer. A terminal
        # methyl or carbonyl oxygen cannot extend the open two-carbon acid
        # skeleton; longer carbon branches require their own parent proof.
        # This does not decide the ring writer's preferred spelling.
        if atom.IsInRing() and atom.GetIsAromatic():
            from .component_mapping import mapped_branch_ring_atoms

            ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
            witnessed = mapped_branch_ring_atoms(mol, Chem, root, host, text, atoms)
            nonring = atoms - ring_atoms
            attachment = mol.GetBondBetweenAtoms(root, host)
            if (ring_atoms <= witnessed and attachment is not None
                    and attachment.GetBondTypeAsDouble() == 1
                    and all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6, 7, 8, 16}
                            and mol.GetAtomWithIdx(i).GetIsAromatic()
                            and not mol.GetAtomWithIdx(i).GetFormalCharge()
                            and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                            for i in ring_atoms)
                    and all(_retained_ring_decoration_proved(
                        mol, i, ring_atoms, atoms, inventory) for i in nonring)):
                return 'compound', ('P-29.3.4', 'P-16.5.1.2',
                                    'P-16.5.1.3.2'), text
        if atom.IsInRing():
            # A fully mapped one-cut cyclic substituent cannot lengthen the
            # acyclic acetate/acetic-acid parent through its ring atoms.
            # Unlike a text-only prefix, the mapped witness covers every
            # physical ring even when it has additional side chains.
            from .component_mapping import mapped_branch_ring_atoms

            ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
            witnessed = mapped_branch_ring_atoms(mol, Chem, root, host, text, atoms)
            boundary = {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms}
            attachment = mol.GetBondBetweenAtoms(root, host)
            if (ring_atoms and ring_atoms <= witnessed and boundary == {(root, host)}
                    and attachment is not None and attachment.GetBondTypeAsDouble() == 1
                    and all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6, 7, 8, 9, 16, 17, 35, 53}
                            and not mol.GetAtomWithIdx(i).GetFormalCharge()
                            and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)):
                return 'compound', ('P-29', 'P-14.3.4.6', 'P-16.5.1.3.2'), text
        return None

    if element in {9, 17, 35, 53}:
        expected = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}[element]
        if (len(atoms) == 1 and atom.GetFormalCharge() == 0
                and atom.GetDegree() == 1 and text == expected):
            return "simple", ("P-14.5", "P-16.5.1.3.2"), text
        return None

    if element == 16 and inventory.supported:
        from .hydrazide_components import unsubstituted_hydrazide_prefix
        groups = [group for group in inventory.groups
                  if group.kind == 'hydrazide' and group.center == root
                  and group.owner == host and frozenset(group.atoms) == atoms]
        if len(groups) == 1:
            typed = unsubstituted_hydrazide_prefix(mol, root, host, atoms)
            if typed is not None and typed[0] == text and frozenset(typed[1]) == atoms:
                return 'compound', ('P-66.3.2.1', 'P-16.5.1.3.2'), text
        return None

    if element == 7:
        if (len(atoms) == 1 and atom.GetFormalCharge() == 0
                and atom.GetTotalNumHs() == 2 and text == "amino"):
            return "simple", ("P-29.3.1", "P-16.5.1.3.2"), text
        from .carbamate_components import mapped_alkoxycarbonylamino_proof

        carbamate = (mapped_alkoxycarbonylamino_proof(mol, Chem, root, host)
                     if (inventory.source_frontier.status == 'selected'
                         and len(inventory.principal) == 1
                         and inventory.principal[0].kind == 'acid') else None)
        if carbamate is not None and carbamate.atoms == atoms and carbamate.text == text:
            return 'compound', (carbamate.rule, 'P-16.5.1.2',
                                'P-16.5.1.3.2'), text
        if _methyl_cycloamino_branch(mol, Chem, root, host, atoms, text, inventory):
            return 'compound', ('P-62.2.3', 'P-16.5.1.2',
                                'P-16.5.1.3.2'), text
        oxygens = [mol.GetAtomWithIdx(i) for i in atoms
                   if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
        if (text == "nitro" and len(atoms) == 3 and atom.GetFormalCharge() == 1
                and len(oxygens) == 2
                and all(o.GetFormalCharge() in {0, -1} for o in oxygens)
                and sorted(mol.GetBondBetweenAtoms(root, o.GetIdx()).GetBondTypeAsDouble()
                           for o in oxygens) == [1.0, 2.0]
                and sorted(o.GetFormalCharge() for o in oxygens) == [-1, 0]):
            return "simple", ("P-29.3.1", "P-16.5.1.3.2"), text
        return None

    if element != 8 or atom.GetFormalCharge() != 0:
        return None
    internal = [n.GetIdx() for n in atom.GetNeighbors()
                if n.GetIdx() in atoms and n.GetAtomicNum() > 1]
    external = [n.GetIdx() for n in atom.GetNeighbors()
                if n.GetIdx() not in atoms and n.GetAtomicNum() > 1]
    if external != [host]:
        return None

    if not internal and len(atoms) == 1 and atom.GetTotalNumHs() == 1 and text == "hydroxy":
        return "simple", ("P-61.1.1", "P-16.5.1.3.2"), text
    if len(internal) != 1:
        return None
    attachment = internal[0]
    attachment_atom = mol.GetAtomWithIdx(attachment)
    bridge = mol.GetBondBetweenAtoms(root, attachment)
    if bridge is None or bridge.GetBondTypeAsDouble() != 1:
        return None

    # P-65.6.3.2.3 acyloxy: root O is the ester bridge oxygen and the complete
    # mapped branch includes the corresponding acid-derived acyl centre.
    ester_groups = [g for g in inventory.groups if g.kind == "ester"
                    and g.ligand == root and g.center == attachment]
    if ester_groups:
        group = ester_groups[0]
        oxygens = [i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
        carbonyl_oxygens = [i for i in oxygens
                            if mol.GetBondBetweenAtoms(group.center, i) is not None
                            and mol.GetBondBetweenAtoms(group.center, i).GetBondTypeAsDouble() == 2]
        carbons = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
        if (group.atoms and set(group.atoms) <= atoms and root in atoms
                and len(oxygens) == 2 and len(carbonyl_oxygens) == 1
                and all(mol.GetAtomWithIdx(i).GetFormalCharge() == 0 for i in oxygens)
                and len(carbons) >= 1
                and _acyloxy_carbon_skeleton(mol, carbons, group.center)):
            benzoyl = (len(carbons) == 7 and all(
                mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons - {group.center}
            ))
            acyl_rule = ("P-65.1.7.2.1" if len(carbons) == 2 or benzoyl
                         else "P-65.1.7.4.1")
            return "compound", ("P-65.6.3.2.3", acyl_rule,
                                 "P-16.5.1.1", "P-16.5.1.3.2"), text
        return None

    substituted = _substituted_short_alkoxy(
        mol, Chem, root, host, atoms, text, inventory)
    if substituted is not None:
        return substituted

    if attachment_atom.IsInRing() and attachment_atom.GetIsAromatic():
        # An aryloxy branch cannot extend the acetic carbon chain through O.
        # Rebuild its entire ring-side substituent and require the mapped
        # writer to witness every cyclic atom before classifying its prefix.
        from .component_mapping import mapped_branch, mapped_branch_ring_atoms
        from .formatting import alkyl_to_alkoxy

        ring_text, ring_atoms = mapped_branch(mol, Chem, attachment, root)
        ring_atoms = frozenset(ring_atoms)
        witnessed = mapped_branch_ring_atoms(mol, Chem, attachment, root, ring_text, ring_atoms)
        cyclic = {i for i in ring_atoms if mol.GetAtomWithIdx(i).IsInRing()}
        if (ring_text and ring_atoms | {root} == atoms and cyclic <= witnessed
                and all(not mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)
                and text == alkyl_to_alkoxy(ring_text)):
            nature = 'simple' if text == 'phenoxy' else 'compound'
            return nature, ('P-63.2.2.1.1', 'P-16.5.1.3.2'), text

    # P-63.2.2.1.1 supplies systematic R–O– prefix construction and calls
    # those concatenated forms compound; the specifically retained simple
    # prefixes are the n-C1–C4 alkoxy groups, phenoxy, and tert-butoxy.
    if attachment_atom.GetAtomicNum() != 6 or any(
            mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in atoms):
        return None
    carbon_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    if not _connected_carbon_component(mol, carbon_atoms, attachment):
        return None
    carbon_text, carbon_atoms_mapped = _mapped_carbon_substituent(
        mol, Chem, attachment, root
    )
    if not carbon_text or frozenset(carbon_atoms_mapped) != frozenset(carbon_atoms):
        return None
    simple_retained = _retained_simple_oxy_graph(mol, carbon_atoms, attachment)
    if simple_retained:
        expected = _retained_oxy_name(mol, carbon_atoms, attachment)
        if not expected or text != expected:
            return None
        return "simple", ("P-63.2.2.2", "P-16.5.1.3.2"), expected

    # Source-derived systematic R–O prefixes are compound prefixes. Keep this
    # branch to unmodified saturated acyclic carbon trees; each rooted carbon
    # substituent must still have a complete mapped name matching the source
    # R′ + oxy construction.
    adjacency = {i: set() for i in carbon_atoms}
    for i in carbon_atoms:
        if mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetIsAromatic():
            return None
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j in carbon_atoms:
                if mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1:
                    return None
                adjacency[i].add(j)
    if (sum(map(len, adjacency.values())) // 2 != len(carbon_atoms) - 1
            or any(len(neighbors) > 2 for neighbors in adjacency.values())):
        return None
    root_degree = len(adjacency[attachment])
    if root_degree not in {1, 2, 3}:
        return None
    # P-63.2.2.1.1: systematic prefixes combine an organyl component with
    # oxy. The source-retained prefix cases were handled above. Other alkyl
    # forms are compound prefixes; enclosure is needed for an internally
    # attached or branched/locanted organyl component. Verify, rather than
    # repair, the existing mapped writer's exact spelling.
    locanted_or_branched = (
        root_degree != 1
        or any(len(neighbors) > 2 for neighbors in adjacency.values())
        or any(character.isdigit() for character in carbon_text)
    )
    component_name = f"({carbon_text})oxy" if locanted_or_branched else carbon_text + "oxy"
    if text != component_name:
        return None
    return "compound", ("P-63.2.2.1.1", "P-16.5.1.1", "P-16.5.1.3.2"), component_name


def _substituted_short_alkoxy(mol, Chem, root, host, atoms, text, inventory):
    """Prove a substituted retained alkoxy unit and its compound nature.

    The acetic parent has already supplied the P-14.3.4.6 locant proof. This
    narrower P-63.2.2.2/P-16.3.5 extension accepts only a unique senior acid
    or ester and a completely mapped short alkoxy branch with terminal
    hydroxy/halogen or further acyclic ether units. Other branch and stereo
    states delegate.
    """
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind not in {'acid', 'carboxylate', 'ester'}
            or inventory.principal[0].owner != host):
        return None
    atoms = frozenset(atoms)
    carbons = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    substituents = atoms - carbons - {root}
    if not carbons or not substituents:
        return None
    for idx in substituents:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() not in {8, 9, 17, 35, 53}
                or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.IsInRing()
                or any(n.GetIdx() not in carbons for n in atom.GetNeighbors())
                or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return None
        if atom.GetAtomicNum() == 8:
            if (atom.GetDegree(), atom.GetTotalNumHs()) not in {(1, 1), (2, 0)}:
                return None
        elif atom.GetDegree() != 1 or atom.GetTotalNumHs() != 0:
            return None
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms):
        return None
    for group in inventory.groups:
        overlap = set(group.atoms) & atoms
        if overlap and (group.kind != 'alcohol' or not set(group.atoms) <= atoms
                        or group.seniority <= inventory.principal[0].seniority):
            return None
    for item in Chem.FindPotentialStereo(mol):
        if item.type == Chem.StereoType.Atom_Tetrahedral and item.centeredOn in atoms:
            return None
        if item.type == Chem.StereoType.Bond_Double:
            bond = mol.GetBondWithIdx(item.centeredOn)
            if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} & atoms:
                return None
    from .branch.contracted_alkoxy import mapped_contracted_alkoxy

    mapped = mapped_contracted_alkoxy(mol, Chem, root, host)
    if mapped is None or mapped[0] != text or mapped[1] != atoms:
        return None
    return 'compound', ('P-63.2.2.2', 'P-16.3.5', 'P-16.5.1.1'), text


def _methyl_cycloamino_branch(mol, Chem, root, host, atoms, text, inventory):
    """Prove the amino unit and each child, including its mapped ring.

    This is a neutral tertiary N bearing methyl and a single unsubstituted,
    saturated carbon/oxygen monocycle. Other N ligands or ring decorations
    are deliberately deferred to their own component evidence.
    """
    atom = mol.GetAtomWithIdx(root)
    amines = [g for g in inventory.groups if g.kind == 'amine' and g.center == root]
    if (not inventory.supported or len(amines) != 1 or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons() or atom.IsInRing() or atom.GetDegree() != 3
            or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())
            or mol.GetBondBetweenAtoms(root, host) is None):
        return False
    children = [n for n in atom.GetNeighbors() if n.GetIdx() != host]
    if len(children) != 2 or any(n.GetAtomicNum() != 6 for n in children):
        return False
    methyls = [n for n in children if n.GetDegree() == 1 and n.GetTotalNumHs() == 3]
    ring_roots = [n for n in children if n.IsInRing() and not n.GetIsAromatic()]
    if len(methyls) != 1 or len(ring_roots) != 1:
        return False
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms
    from .prefix_sequences import mononuclear_prefix_sequence

    methyl = methyls[0].GetIdx()
    ring_root = ring_roots[0].GetIdx()
    methyl_text, methyl_owned = mapped_branch(mol, Chem, methyl, root)
    ring_text, ring_owned = mapped_branch(mol, Chem, ring_root, root)
    if (methyl_text != 'methyl' or methyl_owned != {methyl} or not ring_text
            or not 3 <= len(ring_owned) <= 7
            or set(atoms) != {root} | set(methyl_owned) | set(ring_owned)
            or set(methyl_owned) & set(ring_owned)):
        return False
    if (not set(ring_owned) <= mapped_branch_ring_atoms(
            mol, Chem, ring_root, root, ring_text, ring_owned)
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8}
                   or not mol.GetAtomWithIdx(i).IsInRing()
                   or mol.GetAtomWithIdx(i).GetIsAromatic()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in ring_owned)
            or sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in ring_owned) > 1
            or any(b.GetBondTypeAsDouble() != 1 for i in ring_owned
                   for b in mol.GetAtomWithIdx(i).GetBonds())):
        return False
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()
             if set(ring) & set(ring_owned)]
    if rings != [set(ring_owned)]:
        return False
    sequence = mononuclear_prefix_sequence(((methyl_text, 'simple'), (ring_text, 'simple')))
    return sequence is not None and sequence.text + 'amino' == text


def _connected_carbon_component(mol, carbons, start):
    """Prove the acyl/organyl carbon subset is connected to its attachment."""
    if not carbons or start not in carbons:
        return False
    seen, todo = set(), [start]
    while todo:
        idx = todo.pop()
        if idx in seen:
            continue
        seen.add(idx)
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = neighbor.GetIdx()
            if other in carbons and other not in seen:
                todo.append(other)
    return seen == set(carbons)


def _acyloxy_carbon_skeleton(mol, carbons, carbonyl):
    """Admit source-closed straight alkanoyl and benzoyl prefix skeletons."""
    if not _connected_carbon_component(mol, carbons, carbonyl):
        return False
    if len(carbons) == 7:
        ring_carbons = carbons - {carbonyl}
        if (len(ring_carbons) == 6
                and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_carbons)):
            internal = [n.GetIdx() for n in mol.GetAtomWithIdx(carbonyl).GetNeighbors()
                        if n.GetIdx() in ring_carbons]
            edges = [b for b in mol.GetBonds()
                     if b.GetBeginAtomIdx() in ring_carbons
                     and b.GetEndAtomIdx() in ring_carbons]
            return len(internal) == 1 and len(edges) == 6 and all(b.GetIsAromatic() for b in edges)
    if any(mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons):
        return False
    adjacency = {i: set() for i in carbons}
    for i in carbons:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j not in carbons:
                continue
            if mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1:
                return False
            adjacency[i].add(j)
    return (
        len(adjacency[carbonyl]) == 1
        and all(len(neighbors) <= 2 for neighbors in adjacency.values())
        and sum(map(len, adjacency.values())) // 2 == len(carbons) - 1
        and len([i for i, neighbors in adjacency.items() if len(neighbors) == 1]) == 2
    )


def _mapped_carbon_substituent(mol, Chem, root, blocked):
    from .component_mapping import mapped_branch
    return mapped_branch(mol, Chem, root, blocked)


def _retained_simple_oxy_graph(mol, carbons, attachment):
    """Recognize P-63's source-retained simple oxy groups by graph shape."""
    if _tert_butoxy_graph(mol, carbons, attachment):
        return True
    if len(carbons) in {1, 2, 3, 4}:
        adjacency = {i: set() for i in carbons}
        for i in carbons:
            for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
                if neighbor.GetIdx() in carbons:
                    if mol.GetBondBetweenAtoms(i, neighbor.GetIdx()).GetBondTypeAsDouble() != 1:
                        return False
                    adjacency[i].add(neighbor.GetIdx())
        if (all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons)
                and len(adjacency[attachment]) <= 1
                and all(len(neighbors) <= 2 for neighbors in adjacency.values())
                and sum(map(len, adjacency.values())) // 2 == len(carbons) - 1):
            return True
    # Plain phenoxy is explicitly retained.
    if len(carbons) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons):
        edges = [b for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in carbons and b.GetEndAtomIdx() in carbons]
        return len(edges) == 6 and all(b.GetIsAromatic() for b in edges)
    return False


def _retained_oxy_name(mol, carbons, attachment):
    """Return P-63's retained simple oxy-prefix spelling from graph shape."""
    if _tert_butoxy_graph(mol, carbons, attachment):
        return "tert-butoxy"
    if len(carbons) in {1, 2, 3, 4}:
        names = {1: "methoxy", 2: "ethoxy", 3: "propoxy", 4: "butoxy"}
        return names[len(carbons)]
    if len(carbons) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons):
        return "phenoxy"
    return None


def _retained_ring_decoration_proved(mol, idx, ring_atoms, branch_atoms, inventory):
    """Prove a terminal decoration that cannot extend the acid carbon chain."""
    atom = mol.GetAtomWithIdx(idx)
    if (atom.GetDegree() != 1 or atom.GetFormalCharge() or atom.GetIsotope()
            or atom.GetNumRadicalElectrons()):
        return False
    owner = atom.GetNeighbors()[0].GetIdx()
    if owner not in ring_atoms:
        return False
    bond_order = atom.GetBonds()[0].GetBondTypeAsDouble()
    if bond_order == 1:
        return (atom.GetAtomicNum() in {9, 17, 35, 53}
                or (atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
                    and atom.GetTotalNumHs() == 3))
    if atom.GetAtomicNum() != 8 or bond_order != 2 or not inventory.supported:
        return False
    # P-41 places the carboxylic acid above a ring lactam. Bind the terminal
    # oxygen to exactly one source-recognized lactam wholly owned by this
    # branch; an unclassified carbonyl needs its own functional proof.
    groups = [g for g in inventory.groups if g.kind == 'lactam'
              and g.center == owner and idx in g.atoms
              and set(g.atoms) <= (set(ring_atoms) | {idx})
              and set(g.atoms) <= set(branch_atoms)]
    return len(groups) == 1


def _tert_butoxy_graph(mol, carbons, attachment):
    """Recognize R-O-tert-butyl from its tertiary attachment carbon."""
    if len(carbons) != 4 or attachment not in carbons:
        return False
    atom = mol.GetAtomWithIdx(attachment)
    if atom.IsInRing() or atom.GetIsAromatic():
        return False
    neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in carbons]
    if len(neighbors) != 3:
        return False
    if any(mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetIsAromatic() for i in carbons):
        return False
    return all(
        sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in carbons) == 1
        for i in carbons - {attachment}
    )
