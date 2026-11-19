"""Mapped multi-site acyl-halide functional-class construction (P-65.5)."""

from collections import Counter

from .component_mapping import MappedComponent, decorate_component, fully_covered
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


_HALIDES = {9: "fluoride", 17: "chloride", 35: "bromide", 53: "iodide"}
_CLASS_ORDER = {name: i for i, name in enumerate(("bromide", "chloride", "fluoride", "iodide"))}
_MULTIPLIERS = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
                7: "hepta", 8: "octa", 9: "nona", 10: "deca"}


def _class_terms(halogen_numbers):
    counts = Counter(_HALIDES[number] for number in halogen_numbers)
    terms = []
    for class_name in sorted(counts, key=lambda name: _CLASS_ORDER[name]):
        count = counts[class_name]
        prefix = _MULTIPLIERS.get(count)
        if count > 1 and prefix is None:
            return None
        terms.append((prefix or "") + class_name)
    return " ".join(terms)


def _acyl_name(acid_name):
    """Apply the P-65.1.7 acyl suffix operations to a selected acid name."""
    if acid_name.endswith("carboxylic acid"):
        return acid_name[:-len("carboxylic acid")] + "carbonyl"
    if acid_name.endswith("oic acid"):
        return acid_name[:-len("oic acid")] + "oyl"
    if acid_name.endswith("ic acid"):
        return acid_name[:-len("ic acid")] + "yl"
    return ""


def _site_halides(mol, center):
    atom = mol.GetAtomWithIdx(center)
    return tuple(sorted(n.GetIdx() for n in atom.GetNeighbors()
                        if n.GetAtomicNum() in _HALIDES
                        and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 1))


def _carbonyl_dihalide(mol, Chem, inventory, atoms):
    """The explicitly covered carbonic-acid carbonyl dihalide cases (P-65.5.3.1)."""
    if len(inventory.principal) != 1 or inventory.principal[0].kind != "acid-halide":
        return None
    center = inventory.principal[0].center
    halides = _site_halides(mol, center)
    oxygens = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
               if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(center, n.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(halides) != 2 or len(oxygens) != 1 or atoms != frozenset({center, *halides, *oxygens}):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           for i in (*halides, *oxygens, center)):
        return None
    terms = _class_terms([mol.GetAtomWithIdx(i).GetAtomicNum() for i in halides])
    if not terms:
        return None
    return MethodDecision("carbonyl " + terms, "functional-class", frozenset({center}), atoms,
                          "P-65.5.3.1",
                          component_roles=(("carbonyl", frozenset({center, *oxygens}), ()),
                                           ("halide", frozenset(halides), ())))


def polyacyl_halide_decision(mol, Chem):
    """Name mapped neutral poly-acyl halides where every acid site is halogenated.

    The capped graph exists only for acid-parent selection. The final component
    decoration is rebuilt on the caller's original graph to preserve original
    atom identity and CIP priorities.
    """
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                    for a in mol.GetAtoms()):
        return None
    # Ring-center CIP can change when the hydroxy cap is replaced by the real
    # halide. Keep the same conservative boundary as the mapped acid-component
    # builder until the capped-to-original ring descriptor assignment is proved.
    if any(a.IsInRing() and int(a.GetChiralTag()) for a in mol.GetAtoms()):
        return None
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.groups:
        return None
    # A free acid or any other senior class owns the principal suffix route;
    # P-65.5.4 expresses the acyl halide as a prefix in that context.
    if not inventory.principal or any(g.kind != "acid-halide" for g in inventory.principal):
        return None
    # P-65.5.1 is an operation on ALL corresponding acid groups, including
    # mixed halides. P-41's leaving-group seniority selects a site in other
    # contexts; it must not remove the other halide components of this
    # functional-class construction (the printed mixed-halide example).
    groups = tuple(g for g in inventory.groups if g.kind == 'acid-halide')
    from .group_priority import compare_acid_priority
    facts = {id(group): fact for group, fact in
             zip(inventory.groups, inventory.source_group_facts)}
    principal_fact = facts.get(id(inventory.principal[0]))
    if (principal_fact is None or any(
            (fact := facts.get(id(group))) is None
            or compare_acid_priority(principal_fact.corresponding_acid,
                                     fact.corresponding_acid).status != 'tie'
            for group in groups)):
        return None

    sites = []
    all_halides = []
    for group in groups:
        halides = _site_halides(mol, group.center)
        if not halides:
            return None
        sites.append((group.center, halides))
        all_halides.extend(halides)

    # A single acyl site with two halides is covered only by the explicit
    # carbonyl dihalide pattern. Ordinary mono-acyl halides keep their route.
    if len(sites) == 1:
        return _carbonyl_dihalide(mol, Chem, inventory,
                                  frozenset(a.GetIdx() for a in mol.GetAtoms()
                                            if a.GetAtomicNum() > 1))
    if len(all_halides) != len(set(all_halides)):
        return None
    if any(len(halides) != 1 for _, halides in sites):
        return None
    # The exact class terms are based on original halogen atoms. An isotope on
    # a halide or oxygen has no proved P-82.6.4 placement in this construction.
    if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           for i in all_halides):
        return None
    if any(a.GetAtomicNum() == 8 and a.GetIsotope() for a in mol.GetAtoms()):
        return None

    rw = Chem.RWMol(mol)
    for idx in all_halides:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetAtomicNum(8)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None

    # Parent selection uses the ordinary capped graph. Output stereochemistry
    # is deliberately never inferred from this graph.
    parent_graph = Chem.Mol(capped)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    from .cross_route_selection import cross_route_selection_outcome
    outcome = cross_route_selection_outcome(parent_graph, Chem)
    centers = frozenset(center for center, _ in sites)
    issues = ()
    if outcome.selection is not None:
        parent = outcome.selection.selected
    elif outcome.status == 'pending':
        # An unavailable preferred-parent proof is not evidence against a
        # complete mapped ring construction. Preserve that general route,
        # carrying the original unresolved obligation into the public result.
        from .mapped_ring_parents import records
        parent_groups = selection_inventory(parent_graph).principal
        alternatives = [record for record in records(parent_graph, Chem, principal=parent_groups)
                        if record.principal_centers == centers
                        and record.construction is not None
                        and fully_covered(parent_graph, record.construction.atoms,
                                          record.construction.embedded_atoms)]
        if not alternatives:
            return None
        parent = min(alternatives, key=lambda record: record.name)
        issues = (SelectionLimitation('acid-derivative-method', 'P-44/P-45/P-65.5.1',
                                      outcome.reason),)
    else:
        return None
    if parent.principal_centers != centers:
        # Every halide class cited below must belong to a suffix acid site in
        # the corresponding selected parent. A partial parent cannot silently
        # acquire the all-replaced functional-class construction.
        return None
    construction = parent.construction
    if construction is None or not construction.parent.endswith("acid"):
        return None
    class_names = _class_terms([mol.GetAtomWithIdx(i).GetAtomicNum() for i in all_halides])
    if not class_names:
        return None

    # Remove temporary hydroxy-cap atoms from the mapped acid component and
    # translate the remaining typed map directly to original atom indices.
    halide_set = frozenset(all_halides)
    mapped = MappedComponent(
        construction.parent,
        tuple((idx, loc) for idx, loc in construction.atom_locants if idx not in halide_set),
        frozenset(set(construction.atoms) - halide_set),
        construction.prefixes,
        frozenset(set(construction.embedded_atoms) - halide_set),
        construction.hydro_prefix,
        construction.systematic_parent,
        construction.nitrogen_site_locants,
        construction.isotope_unlocanted_elements,
        construction.prefix_mode,
        frozenset(set(construction.local_isotope_atoms) - halide_set),
        construction.omit_methyl_prefix_locant,
    )
    # The construction's typed writer rendered the acid form above. Re-render
    # its mapped component on the source graph, then derive only the class
    # ending; this retains mapped prefixes and original-graph CIP labels.
    acid_name = decorate_component(mol, Chem, mapped)
    if not acid_name:
        return None
    acyl_name = _acyl_name(acid_name)
    if not acyl_name:
        return None
    covered = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if set(all_halides) & set(mapped.atoms):
        return None
    centers = frozenset(center for center, _ in sites)
    return MethodDecision(acyl_name + " " + class_names, "functional-class", centers,
                          covered, "P-65.5.1/P-65.1.7.1/P-65.1.7.2/P-65.1.7.4",
                          tuple((idx, loc) for idx, loc in mapped.atom_locants),
                          component_roles=(("acyl", covered - halide_set,
                                            tuple((idx, loc) for idx, loc in mapped.atom_locants)),
                                           ("halide", halide_set, ())),
                          preference_issues=issues)
