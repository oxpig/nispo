"""P-51.2/P-65 functional-class construction with original component ownership."""
from dataclasses import dataclass, replace
from functools import lru_cache

from .molecule_iteration import iter_atoms
from .branch_rules import collect_subtree
from .extended_inventory import selection_inventory
from .formatting import prefix_alphabetization_key
from .component_mapping import MappedComponent, decorate_component, has_annotations
from .naming_methods import MethodDecision
from .oxygen_isotope_suffixes import (
    suffix_oxygen_descriptors, organyl_oxygen_qualifier,
    apply_organyl_oxygen_qualifier,
)
from .isotope_component_coverage import component_record_from_atom_map
from .isotope_component_coverage import IsotopeComponentRecord
from .ester_isotope_coverage import (
    EsterOxygenCoverageResult, EsterOxygenRoles,
    render_ester_acid_component_with_coverage,
    validate_ester_oxygen_coverage,
)


@dataclass(frozen=True,slots=True)
class AcidComponent:
    name: str
    atoms: frozenset[int]
    atom_locants: tuple
    identity: str
    substituted: bool = False
    ester_organyl_marker: str = ""
    carbonyl_oxygen: int | None = None
    bridge_oxygen: int | None = None
    suffix_oxygen_isotopes: tuple = ()
    construction: MappedComponent | None = None


@dataclass(frozen=True, slots=True)
class EsterIsotopeMethodDecision(MethodDecision):
    """Single-ester decision carrying its validated original-atom audit facts."""

    isotope_coverage: EsterOxygenCoverageResult | None = None
    isotope_components: tuple[IsotopeComponentRecord, ...] = ()


@lru_cache(maxsize=512)
def _capped_acid_selection(Chem, graph_binary, select_parent):
    """Share complete evidence for an exact temporary acid-parent graph.

    Callers have already capped or neutralized their component. The binary
    includes all native properties, atom order, isotope labels and naming
    modes; the backend and selector are also part of the key. Only immutable
    selection evidence escapes the private graph. Original component mapping,
    numbering and decoration still run separately for every source molecule.
    """
    return select_parent(Chem.Mol(graph_binary), Chem)


@lru_cache(maxsize=512)
def _acid_parent_proposal(Chem, graph_binary, select_parent):
    """Reuse exact capped-fragment evidence, never whole-molecule names.

    Native properties, atom/bond order and all annotations belong to the key.
    Reconstruct a private graph so the cache retains only the immutable
    proposal; local locants are translated back by each derivative writer.
    Original-graph decoration and isotope-role validation remain uncached.
    """
    graph = Chem.Mol(graph_binary)
    inventory = selection_inventory(graph)
    if len(inventory.principal) != 1 or inventory.principal[0].kind != 'acid':
        return None
    selection = _capped_acid_selection(Chem, graph_binary, select_parent)
    return selection.selected if selection is not None else None


def _unsaturated_carbon_tree_acid_parent(acid, Chem):
    """Prove an otherwise deferred monoacid parent on a carbon tree.

    The ordinary cross-route selector intentionally leaves many single-site
    acid proposals pending.  An unannotated hydrocarbon acid component has a
    smaller, complete graph frontier: P-44 compares every eligible acid-bearing
    carbon path by length and multiple bonds before its numbering is rendered.
    This fallback uses that frontier only, with no additional heteroatom or
    stereochemical roles whose competing parent evidence it cannot prove.
    """
    if (acid.GetStereoGroups() or acid.GetRingInfo().NumRings()
            or any(a.GetAtomicNum() not in {6, 8} or a.GetFormalCharge()
                   or a.GetNumRadicalElectrons() or a.GetIsotope()
                   or int(a.GetChiralTag()) for a in iter_atoms(acid))
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in acid.GetBonds())):
        return None
    inventory = selection_inventory(acid)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'acid'):
        return None
    group = inventory.principal[0]
    if ({a.GetIdx() for a in iter_atoms(acid) if a.GetAtomicNum() == 8}
            != set(group.atoms) - {group.center}):
        return None
    carbon = {a.GetIdx() for a in iter_atoms(acid) if a.GetAtomicNum() == 6}
    if not any(b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon
               and b.GetBondTypeAsDouble() > 1 for b in acid.GetBonds()):
        return None
    from .suffix_coverage_frontier import maximum_suffix_coverage
    from .complete_functional_parent import _functional_suffix_coverage_frontier

    coverage = maximum_suffix_coverage(acid, Chem, (group,), carbon)
    if coverage.status != 'proved':
        return None
    skipped = set(group.atoms) - {group.center}
    frontier = _functional_suffix_coverage_frontier(
        acid, Chem, (group,), 'acid', coverage, skipped, [])
    if (frontier.status not in {'complete', 'tie'} or not frontier.candidates
            or len({candidate.name for candidate in frontier.candidates}) != 1):
        return None
    return frontier.candidates[0]


def _acid_component(mol,Chem,center,leaving,*,allow_mapped_ring=False,allow_mapped_chain=False):
    atoms=frozenset(collect_subtree(mol,center,leaving))
    if leaving in atoms:return None
    bridge=mol.GetAtomWithIdx(leaving)
    if bridge.GetIsotope() and bridge.GetAtomicNum()!=8:return None
    if int(bridge.GetChiralTag()):return None
    carbonyl_oxygens=[a.GetIdx() for a in mol.GetAtomWithIdx(center).GetNeighbors()
                      if a.GetAtomicNum()==8 and
                      mol.GetBondBetweenAtoms(center,a.GetIdx()).GetBondTypeAsDouble()==2]
    if len(carbonyl_oxygens)!=1:return None
    carbonyl_oxygen=carbonyl_oxygens[0]
    carbonyl_atom=mol.GetAtomWithIdx(carbonyl_oxygen)
    if carbonyl_atom.GetIsotope() and carbonyl_atom.GetAtomicNum()!=8:return None
    if int(carbonyl_atom.GetChiralTag()):return None
    order=sorted(atoms);graph=Chem.RWMol(mol)
    for i in reversed(range(mol.GetNumAtoms())):
        if i not in atoms:graph.RemoveAtom(i)
    cap=Chem.Atom(8)
    # The capped acid's hydroxy oxygen represents the original ester/anhydride
    # bridge oxygen for isotope identity and acid-side suffix counting.
    cap.SetIsotope(bridge.GetIsotope())
    oxygen=graph.AddAtom(cap);graph.AddBond(order.index(center),oxygen,Chem.BondType.SINGLE)
    acid=graph.GetMol()
    try:Chem.SanitizeMol(acid)
    except (ValueError,RuntimeError):return None
    # Parent selection uses the unmodified acid ending. Oxygen labels are
    # assembled by the typed ester writer below; carbon labels remain in place
    # so their locants still participate in parent selection.
    parent_graph=Chem.Mol(acid)
    parent_graph.GetAtomWithIdx(order.index(carbonyl_oxygen)).SetIsotope(0)
    parent_graph.GetAtomWithIdx(len(order)).SetIsotope(0)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph,cleanIt=True,force=True)
    # Cross-route scope is the authoritative bounded eligibility proof for
    # subordinate groups, including mapped sulfur characteristic prefixes.
    # Ring stereochemistry is numbered against the acid component as capped;
    # its CIP comparison can differ after reconnection to the real derivative.
    if any(mol.GetAtomWithIdx(i).IsInRing() and
           int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms):return None
    from .cross_route_selection import cross_route_selection
    selected = _acid_parent_proposal(
        Chem, parent_graph.ToBinary(Chem.PropertyPickleOptions.AllProps),
        cross_route_selection)
    if selected is None:
        selected = _unsaturated_carbon_tree_acid_parent(acid, Chem)
    if selected is None:
        from .bounded_retained_carboxylic_parents import (
            amino_acetic_acid_parent, aromatic_acetic_acid_parent,
            substituted_benzoic_acid_parent,
        )

        selected = (substituted_benzoic_acid_parent(acid, Chem)
                    or aromatic_acetic_acid_parent(acid, Chem)
                    or amino_acetic_acid_parent(acid, Chem))
    if selected is None and (allow_mapped_ring or bridge.GetAtomicNum() in {9, 17, 35, 53}):
        # The leaving group remains outside the capped ring acid. Compare
        # mapped ring constructions with full ownership of the acid residue.
        from .mapped_ring_parents import records

        rows = records(parent_graph, Chem, general_branches=True)
        if len(rows) == 1 and rows[0].principal_centers == {order.index(center)}:
            selected = rows[0]
    if selected is None and (allow_mapped_chain or bridge.GetAtomicNum() in {9, 17, 35, 53}):
        from .mapped_principal_general import _carbon_proposals
        from .parent_selection import parent_frontier

        inventory = selection_inventory(parent_graph)
        if (inventory.supported and len(inventory.principal) == 1
                and inventory.principal[0].kind == "acid"
                and inventory.principal[0].center == order.index(center)):
            rows = list(_carbon_proposals(parent_graph, Chem, inventory.principal, inventory, 0))
            if rows and all(facts is not None for _, facts in rows):
                best = parent_frontier(rows, lambda row: row[1])
                if len({row.name for row, _ in best}) == 1:
                    selected = best[0][0]
    if selected is None:return None
    construction=selected.construction
    if construction is None or not construction.parent.endswith(' acid'):return None
    # Map the selected acid construction back to the original graph. The
    # artificial OH belongs only to parent selection; it is not part of the
    # ester/halide component.
    locants={order[i]:loc for i,loc in construction.atom_locants if i<len(order)}
    owned={order[i] for i in construction.atoms if i<len(order)}
    embedded={order[i] for i in construction.embedded_atoms if i<len(order)}
    # Some proposals expose extra suffix locants. Add only atoms outside the
    # typed embedded characteristic group; embedded carbonyl centers are
    # owned by the acid ending and cannot take skeletal isotope locants here.
    for i,loc in selected.atom_locants:
        if i<len(order) and order[i] not in embedded:
            locants.setdefault(order[i],loc)
            owned.add(order[i])
    translated=MappedComponent(
        construction.parent,
        tuple((idx,loc) for idx,loc in locants.items() if idx in owned),
        frozenset(owned), construction.prefixes, frozenset(embedded),
        construction.hydro_prefix, construction.systematic_parent,
        construction.nitrogen_site_locants, construction.isotope_unlocanted_elements,
        construction.prefix_mode,
        frozenset(order[i] for i in construction.local_isotope_atoms
                  if i < len(order) and order[i] in owned),
        construction.omit_methyl_prefix_locant)
    if translated.parent == "acetic acid":
        from .retained_acid_rules import retained_acid_locant_evidence

        evidence = retained_acid_locant_evidence(
            mol, Chem, parent_class=translated.parent, principal_kind="ester",
            parent_atoms=translated.atoms,
            atom_locants=translated.atom_locants,
            prefixes=translated.prefixes,
            isotope_forces_locants=any(atom.GetIsotope() for atom in iter_atoms(mol)),
        )
        translated = MappedComponent(
            translated.parent, translated.atom_locants, translated.atoms,
            translated.prefixes, translated.embedded_atoms, translated.hydro_prefix,
            translated.systematic_parent, translated.nitrogen_site_locants,
            translated.isotope_unlocanted_elements, translated.prefix_mode,
            translated.local_isotope_atoms, translated.omit_methyl_prefix_locant,
            retained_acid_locant_evidence=evidence)
    if translated.atoms & translated.embedded_atoms:return None
    # A branch included in an ordinary chemical prefix has no independent
    # isotope/stereo construction here. Decline it until a typed child map is
    # available rather than moving its labels onto the skeletal acid parent.
    embedded_unsupported = set(translated.embedded_atoms) - {carbonyl_oxygen}
    if has_annotations(mol,embedded_unsupported):return None
    # A mapped carbonyl oxygen is part of the typed acid suffix. Its isotope
    # is rendered at the acid ending below; no other hidden annotation is
    # admitted by this exception.
    if carbonyl_atom.GetIsotope():
        if carbonyl_oxygen not in translated.atoms:return None
        translated=MappedComponent(
            translated.parent,translated.atom_locants,translated.atoms,
            translated.prefixes,translated.embedded_atoms,translated.hydro_prefix,
            translated.systematic_parent,translated.nitrogen_site_locants,
            translated.isotope_unlocanted_elements,translated.prefix_mode,
            translated.local_isotope_atoms|{carbonyl_oxygen},
            translated.omit_methyl_prefix_locant,
            retained_acid_locant_evidence=translated.retained_acid_locant_evidence)
    mapped_atoms=frozenset(dict(translated.atom_locants))
    if not translated.atoms<=mapped_atoms:return None
    if has_annotations(mol,atoms-mapped_atoms-translated.embedded_atoms):return None
    named=decorate_component(mol,Chem,translated)
    if not named:return None
    oxygen_isotopes=tuple(mass for mass in (carbonyl_atom.GetIsotope(),bridge.GetIsotope()) if mass)
    if oxygen_isotopes:
        named=suffix_oxygen_descriptors(named,oxygen_isotopes)
        if not named:return None
    # Atom-map labels identify source atoms for callers, not different acid
    # species. Keep isotope/configuration identity while removing those
    # arbitrary labels only on the canonical-identity scratch graph.
    identity_graph = Chem.Mol(acid)
    for atom in identity_graph.GetAtoms():
        atom.SetAtomMapNum(0)
    return AcidComponent(named,atoms,translated.atom_locants,
                         Chem.MolToSmiles(identity_graph,isomericSmiles=True),
                         bool(translated.prefixes) or has_annotations(mol,atoms),
                         organyl_oxygen_qualifier(bridge.GetIsotope(),carbonyl_atom.GetIsotope()),
                         carbonyl_oxygen,leaving,
                         oxygen_isotopes,translated)


def _acyl(acid):
    for ending,term in (('acetic acid','acetyl'),('benzoic acid','benzoyl'),('formic acid','formyl')):
        if acid.endswith(ending):return acid[:-len(ending)]+term
    if acid.endswith('carboxylic acid'):return acid[:-len('carboxylic acid')]+'carbonyl'
    if acid.endswith('oic acid'):return acid[:-len('oic acid')]+'oyl'
    return ''


def _anion(acid):
    for ending,term in (('acetic acid','acetate'),('benzoic acid','benzoate'),('formic acid','formate')):
        if acid.endswith(ending):return acid[:-len(ending)]+term
    if acid.endswith('ic acid'):return acid[:-len('ic acid')]+'ate'
    return ''


def acid_derivative_decision(mol,Chem):
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
           for a in iter_atoms(mol)):return None
    inventory=selection_inventory(mol);groups=inventory.principal
    if not groups or groups[0].kind not in {'ester','acid-halide','anhydride'}:return None
    kind=groups[0].kind;atoms=frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1)
    centers=frozenset(g.center for g in groups)
    # P-82.6.4 directly proves these oxygen roles for ordinary esters. Other
    # non-carbon isotope locations and acid-derivative classes stay deferred.
    allowed_oxygen=set()
    for group in groups:
        for atom in mol.GetAtomWithIdx(group.center).GetNeighbors():
            if atom.GetAtomicNum()==8 and mol.GetBondBetweenAtoms(group.center,atom.GetIdx()).GetBondTypeAsDouble()==2:
                allowed_oxygen.add(atom.GetIdx())
        if group.kind=='ester':allowed_oxygen.add(group.ligand)
    if any(a.GetIsotope() and a.GetAtomicNum()!=6 and
           not (kind=='ester' and a.GetAtomicNum()==8 and a.GetIdx() in allowed_oxygen)
           for a in iter_atoms(mol)):return None
    if kind=='anhydride':
        if len(groups)!=2 or groups[0].ligand!=groups[1].ligand:return None
        oxygen=groups[0].ligand
        if mol.GetAtomWithIdx(oxygen).IsInRing():return None
        parts=[_acid_component(mol,Chem,g.center,oxygen) for g in groups]
        if any(p is None for p in parts):return None
        a,b=parts
        if a.atoms&b.atoms or a.atoms|b.atoms|{oxygen}!=atoms:return None
        # Stereo/isotope labels are carried by each mapped acid component's
        # rendered parent name. The identity check is graph-based and includes
        # those labels, so an asymmetric annotated pair is never called
        # symmetric merely because its unlabelled skeletons match.
        names=[_anhydride_acid_name(a.name)] if a.identity==b.identity else sorted(
            [_anhydride_acid_name(a.name),_anhydride_acid_name(b.name)],
            key=prefix_alphabetization_key)
        if any(not n for n in names):return None
        return MethodDecision(' '.join(names)+' anhydride','functional-class',centers,atoms,
            'P-65.7.1' if a.identity==b.identity else 'P-65.7.2',a.atom_locants+b.atom_locants,
            component_roles=(('acid:1',a.atoms,a.atom_locants),('acid:2',b.atoms,b.atom_locants),
                             ('bridge',frozenset({oxygen}),())))
    if kind=='ester' and len(groups)>=2:
        from .polyol_ester_methods import polyol_ester_decision
        decision=polyol_ester_decision(mol,Chem,groups,atoms,centers)
        if (decision and decision.rule.endswith('P-65.6.3.3.3.2')
                and not has_annotations(mol, atoms)):
            from .mapped_ester_general import mapped_ester_decision

            compatible = mapped_ester_decision(mol, Chem)
            if compatible is not None:
                # Keep the selected site and method evidence. This general
                # functional-class spelling owns the other acid arm as a
                # substituent of the ester alcohol component.
                from .naming_result import SelectionLimitation

                issue = SelectionLimitation('delivery', 'P-65.6.3.3.3.2',
                    'The general ester name cites the unlike acid arm as a substituent.')
                return replace(decision, name=compatible.name, atom_locants=compatible.atom_locants,
                               component_roles=compatible.component_roles,
                               preference_issues=(*decision.preference_issues, issue))
        if decision:return decision
        if len(groups) == len(inventory.groups) == 2 and mol.GetRingInfo().NumRings():
            from .mapped_ester_general import mapped_multiplicative_diester

            return mapped_multiplicative_diester(mol, Chem, groups, atoms)
        return None
    if kind=='acid-halide':
        from .polyacyl_halide_methods import polyacyl_halide_decision
        decision=polyacyl_halide_decision(mol,Chem)
        if decision:return decision
    if len(groups)!=1:return None
    group=groups[0]
    if kind == 'ester' and group.owner >= 0 and mol.GetAtomWithIdx(group.owner).IsInRing():
        from .pyridine_ester_methods import pyridine_carboxylate_decision

        pyridine_ester = pyridine_carboxylate_decision(mol, Chem)
        if pyridine_ester is not None:
            return pyridine_ester
    if kind=='acid-halide':
        leaving=[a for a in mol.GetAtomWithIdx(group.center).GetNeighbors()
                 if a.GetAtomicNum() in {9,17,35,53}]
        if len(leaving)!=1:return None
        part=_acid_component(mol,Chem,group.center,leaving[0].GetIdx())
        if part is None or part.atoms|{leaving[0].GetIdx()}!=atoms:return None
        acyl=_acyl(part.name)
        halide={9:'fluoride',17:'chloride',35:'bromide',53:'iodide'}[leaving[0].GetAtomicNum()]
        return MethodDecision(acyl+' '+halide,'functional-class',centers,atoms,'P-65.5.1',part.atom_locants,
            component_roles=(('acid',part.atoms,part.atom_locants),
                             ('halide',frozenset({leaving[0].GetIdx()}),()))) if acyl else None
    oxygen=group.ligand
    part=_acid_component(mol,Chem,group.center,oxygen)
    if part is None:return None
    roots=[a.GetIdx() for a in mol.GetAtomWithIdx(oxygen).GetNeighbors() if a.GetIdx()!=group.center]
    if len(roots)!=1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum()!=6:return None
    from .component_mapping import mapped_branch
    owner,owned=mapped_branch(mol,Chem,roots[0],oxygen)
    if part.ester_organyl_marker and has_annotations(mol, owned):
        # P-82.6.4 establishes the O-/nuclide-O role qualifier separately
        # from the organyl's own nuclide/stereo descriptors. Their combined
        # grammar is not established here; putting 18O into an ethyl isotope
        # descriptor would wrongly claim that O is an atom of that unit.
        return None
    owner=apply_organyl_oxygen_qualifier(owner,part.ester_organyl_marker)
    if not owner or part.atoms&owned or part.atoms|owned|{oxygen}!=atoms:return None
    anion=_anion(part.name)
    if not anion:return None
    oxygen_labeled = bool(mol.GetAtomWithIdx(oxygen).GetIsotope()
                          or mol.GetAtomWithIdx(part.carbonyl_oxygen).GetIsotope())
    if oxygen_labeled:
        # The typed acid writer owns the acid-parent labels and the carbonyl
        # oxygen's local suffix label. The organyl qualifier owns a labeled
        # bridge oxygen; the same bridge is a separately proved P-82.6.4
        # secondary reference from the acid-ending oxygen count.
        acid_write = render_ester_acid_component_with_coverage(
            mol, Chem, part.construction,
            component_id="ester-acid",
            acid_scope=part.atoms,
            bridge_oxygen_id=oxygen,
            carbonyl_oxygen_id=part.carbonyl_oxygen,
        )
        if acid_write is None or acid_write.text != part.name:
            return None
        organyl_scope = frozenset(owned) | {oxygen}
        try:
            organyl_record = component_record_from_atom_map(
                mol,
                component_id="ester-organyl",
                component_text=owner,
                atom_ids=organyl_scope,
                atom_locants=tuple((idx, None) for idx in sorted(organyl_scope)),
                descriptor_atom_ids=({oxygen} if mol.GetAtomWithIdx(oxygen).GetIsotope()
                                     else ()),
                printed_locants={oxygen: None} if mol.GetAtomWithIdx(oxygen).GetIsotope() else {},
            )
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        qualifier_ids = {oxygen} if part.ester_organyl_marker else set()
        secondary_ids = {oxygen} if mol.GetAtomWithIdx(oxygen).GetIsotope() else set()
        coverage = validate_ester_oxygen_coverage(
            mol,
            EsterOxygenRoles(
                acyl_carbon=group.center,
                bridge_oxygen=oxygen,
                carbonyl_oxygen=part.carbonyl_oxygen,
                organyl_carbon=roots[0],
            ),
            (acid_write.coverage, organyl_record),
            organyl_component_id="ester-organyl",
            acid_component_id="ester-acid",
            organyl_qualifier_atom_ids=qualifier_ids,
            suffix_count_atom_ids=acid_write.suffix_count_atom_ids,
            suffix_count_secondary_atom_ids=secondary_ids,
        )
        if not coverage.complete:
            return None
        return EsterIsotopeMethodDecision(
            owner+' '+anion, 'functional-class', centers, atoms,
            'P-65.6.3.2.1/P-65.6.3.3.1', part.atom_locants,
            component_roles=(('acid',part.atoms,part.atom_locants),
                             ('alcohol',frozenset(owned)|{oxygen},())),
            isotope_coverage=coverage,
            isotope_components=(acid_write.coverage, organyl_record),
        )
    return MethodDecision(owner+' '+anion,'functional-class',centers,atoms,'P-65.6.3.2.1/P-65.6.3.3.1',part.atom_locants,
        component_roles=(('acid',part.atoms,part.atom_locants),('alcohol',frozenset(owned)|{oxygen},())))


def _anhydride_acid_name(name):
    """Drop the acid class term while preserving mapped component labels."""
    return name[:-5] if name.endswith(' acid') else ''


def _polyol_ester_decision(mol,Chem,groups,atoms,centers):
    """Compatibility delegate to the separately mapped polyol writer."""
    from .polyol_ester_methods import polyol_ester_decision
    return polyol_ester_decision(mol,Chem,groups,atoms,centers)
