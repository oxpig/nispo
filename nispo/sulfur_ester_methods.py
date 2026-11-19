"""Mapped functional-class names for a bounded family of sulfur acid esters.

The source gate is a neutral monoester R-S(O)x-O-R' (x=1 or 2), with
carbon-only organyl arms and no competing characteristic group. Acid-parent
selection is delegated to the mapped complete-parent frontier on an in-memory
acid-form component; the ester bridge is kept as its own original-atom role.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class SulfurEsterFrontier:
    disposition: str  # complete, pending, unmatched
    source_scope_proved: bool
    decision: MethodDecision | None
    equivalent_atom_locant_maps: tuple[tuple[tuple[int, str], ...], ...]
    rule: str
    reason: str
    general_construction_permitted: bool = False


@dataclass(frozen=True, slots=True)
class SulfurEsterDecision(MethodDecision):
    """Annotated decision with evidence bound to original graph atom IDs."""

    equivalent_atom_locant_maps: tuple[tuple[tuple[int, str], ...], ...] = ()
    acid_component_proofs: tuple = ()
    acid_copy_to_original: tuple[tuple[tuple[int, int], ...], ...] = ()
    organyl_component_proof: object | None = None
    isotope_component_records: tuple = ()
    isotope_coverage: object | None = None
    original_cip_evidence: tuple["SulfurEsterCIPEvidence", ...] = ()


@dataclass(frozen=True, slots=True)
class SulfurEsterCIPEvidence:
    """A tetrahedral carbon descriptor bound to an original component atom."""

    atom_id: int
    component_id: str
    component_locant: str
    descriptor: str


@dataclass(frozen=True, slots=True)
class _SulfurEster:
    center: int
    acid_carbon: int
    bridge_oxygen: int
    oxo_atoms: tuple[int, ...]
    oxidation: int


@dataclass(frozen=True, slots=True)
class SulfurPolyolEsterDecision(MethodDecision):
    """Mapped repeated sulfur-anion ester construction on one carbon polyol."""

    equivalent_polyol_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()
    equivalent_acid_atom_locants: tuple[tuple[tuple[tuple[int, str], ...], ...], ...] = ()
    acid_component_proofs: tuple = ()
    acid_copy_to_original: tuple = ()
    anion_multiplication: object | None = None


def sulfur_polyol_ester_decision(mol, Chem) -> MethodDecision | None:
    """Name identical sulfur-acid anions on one fully esterified carbon chain.

    This is the P-65.6.3.3.3.1 inverse topology to the monoester route:
    one unbranched polyol-derived multivalent organyl component bears two or
    more identical sulfonate or sulfinate anions. Carbon arms and acid-parent
    construction use the bounded Job 199 grammar. Isotopes, stereochemistry,
    unlike anions, cyclic/branched polyols, and unowned functionality decline.
    """
    if (mol is None or mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol, asMols=False)) != 1):
        return None
    from .constants import MULT, ROOTS
    from .acid_anion_multiplication import multiply_acid_anion

    if any(atom.GetAtomicNum() == 1 or atom.GetIsotope()
           or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           or int(atom.GetChiralTag()) for atom in iter_atoms(mol)):
        return None
    if any(bond.GetStereo() or bond.GetBondDir() != Chem.BondDir.NONE
           for bond in iter_bonds(mol)):
        return None
    centers = [atom for atom in iter_atoms(mol) if atom.GetAtomicNum() == 16]
    if len(centers) not in MULT or len(centers) < 2:
        return None
    sites = []
    for sulfur in centers:
        idx = sulfur.GetIdx()
        oxo = tuple(sorted(
            atom.GetIdx() for atom in sulfur.GetNeighbors()
            if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
            and mol.GetBondBetweenAtoms(idx, atom.GetIdx()).GetBondTypeAsDouble() == 2))
        bridges = [atom.GetIdx() for atom in sulfur.GetNeighbors()
                   if atom.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(idx, atom.GetIdx()).GetBondTypeAsDouble() == 1]
        carbons = [atom.GetIdx() for atom in sulfur.GetNeighbors()
                   if atom.GetAtomicNum() == 6
                   and mol.GetBondBetweenAtoms(idx, atom.GetIdx()).GetBondTypeAsDouble() == 1]
        if (len(oxo) not in {1, 2} or len(bridges) != 1 or len(carbons) != 1
                or len(sulfur.GetNeighbors()) != len(oxo) + 2):
            return None
        bridge = mol.GetAtomWithIdx(bridges[0])
        if (bridge.GetDegree() != 2 or bridge.IsInRing()
                or any(neighbor.GetAtomicNum() != 6
                       for neighbor in bridge.GetNeighbors()
                       if neighbor.GetIdx() != idx)):
            return None
        site = _SulfurEster(idx, carbons[0], bridges[0], oxo, len(oxo))
        if not _ordinary_valence(mol, Chem, site):
            return None
        acid_carbons = _carbon_component(mol, site.acid_carbon)
        if acid_carbons is None:
            return None
        sites.append((site, acid_carbons))
    if len({row[0].bridge_oxygen for row in sites}) != len(sites):
        return None

    acid_carbon_union: set[int] = set()
    acid_atom_sets = []
    acid_names = []
    acid_maps = []
    acid_proofs = []
    acid_constructions = []
    acid_identities = []
    for site, carbon_set in sites:
        if acid_carbon_union & set(carbon_set):
            return None
        acid_carbon_union.update(carbon_set)
        acid_atoms = set(carbon_set) | {site.center, site.bridge_oxygen, *site.oxo_atoms}
        acid_atom_sets.append(frozenset(acid_atoms - {site.bridge_oxygen}))
        acid_mol, new_to_old = _acid_form_component(mol, Chem, site, carbon_set)
        if acid_mol is None:
            return None
        from .extended_inventory import selection_inventory
        from .complete_functional_parent import functional_parent_candidates
        from .parent_selection import parent_frontier
        from .component_annotation_evidence import decorate_component_with_evidence

        inventory = selection_inventory(acid_mol)
        kind = "sulfonic-acid" if site.oxidation == 2 else "sulfinic-acid"
        groups = tuple(group for group in inventory.groups if group.kind == kind)
        if len(groups) != 1 or len(inventory.groups) != 1:
            return None
        candidates = functional_parent_candidates(
            acid_mol, Chem, include_single=True, principal=groups)
        if not candidates or any(candidate.parent_selection is None
                                 for candidate in candidates):
            return None
        maximal = parent_frontier(candidates, lambda candidate: candidate.parent_selection)
        if not maximal or any(candidate.construction is None for candidate in maximal):
            return None
        if len({candidate.name for candidate in maximal}) != 1:
            return None
        constructions = tuple(candidate.construction for candidate in maximal)
        if len({(construction.parent, construction.prefixes,
                 construction.systematic_parent)
                for construction in constructions}) != 1:
            return None
        acid_constructions.append(constructions[0])
        decorations = tuple(decorate_component_with_evidence(
            acid_mol, Chem, candidate.construction,
            component_id=f"sulfur-polyol-acid:{len(acid_proofs) + 1}")
            for candidate in maximal)
        if any(decoration is None or decoration.text != maximal[0].name
               for decoration in decorations):
            return None
        acid_name = decorations[0].text
        ending = "sulfonic acid" if site.oxidation == 2 else "sulfinic acid"
        replacement = "sulfonate" if site.oxidation == 2 else "sulfinate"
        if not acid_name.endswith(ending):
            return None
        acid_names.append(acid_name[:-len(ending)] + replacement)
        mapped_frontier = []
        for candidate in maximal:
            rows = []
            for atom_idx, locant in candidate.atom_locants:
                if atom_idx not in new_to_old:
                    return None
                original = new_to_old[atom_idx]
                if original != site.bridge_oxygen:
                    rows.append((original, str(locant)))
            mapped_frontier.append(tuple(sorted(rows)))
        acid_maps.append(tuple(sorted(set(mapped_frontier))))
        acid_proofs.append(tuple(decorations))
        acid_identities.append(Chem.MolToSmiles(acid_mol, canonical=True,
                                                isomericSmiles=True))
        acid_proofs[-1] = (acid_proofs[-1], tuple(sorted(new_to_old.items())))

    if len(set(acid_names)) != 1 or len(set(acid_identities)) != 1:
        return None
    acid_count = len(sites)
    acid_multiplication = multiply_acid_anion(
        acid_names[0], acid_count,
        construction=acid_constructions[0])
    if acid_multiplication is None:
        return None

    bridges = {site.bridge_oxygen for site, _ in sites}
    all_carbons = {atom.GetIdx() for atom in iter_atoms(mol)
                   if atom.GetAtomicNum() == 6}
    framework = all_carbons - acid_carbon_union
    if len(framework) < 2 or not framework:
        return None
    adjacency = {idx: set() for idx in framework}
    for idx in framework:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other in framework:
                if mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1:
                    return None
                adjacency[idx].add(other)
            elif other not in bridges:
                return None
    if (any(len(neighbors) > 2 for neighbors in adjacency.values())
            or sum(map(len, adjacency.values())) // 2 != len(framework) - 1):
        return None
    leaves = tuple(idx for idx, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(leaves) != 2:
        return None
    attachment_by_bridge = {}
    for site, _ in sites:
        bridge = mol.GetAtomWithIdx(site.bridge_oxygen)
        neighbors = [neighbor.GetIdx() for neighbor in bridge.GetNeighbors()
                     if neighbor.GetIdx() != site.center]
        if len(neighbors) != 1 or neighbors[0] not in framework:
            return None
        attachment_by_bridge[site.bridge_oxygen] = neighbors[0]
    attachment_atoms = tuple(attachment_by_bridge.values())
    if (len(set(attachment_atoms)) != len(attachment_atoms)
            or not set(attachment_atoms) <= framework):
        return None
    for idx in framework:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons() or atom.GetTotalValence() != 4:
            return None
        for neighbor in atom.GetNeighbors():
            other = neighbor.GetIdx()
            if other in framework and mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble() != 1:
                return None

    from .constants import ROOTS
    stem = ROOTS.get(len(framework))
    if stem is None:
        return None
    candidates = []
    for start, end in (leaves, tuple(reversed(leaves))):
        path = [start]
        while path[-1] != end:
            next_atoms = adjacency[path[-1]] - set(path)
            if len(next_atoms) != 1:
                return None
            path.append(next(iter(next_atoms)))
        if len(path) != len(framework):
            continue
        mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        site_locs = tuple(sorted(int(mapping[host]) for host in attachment_atoms))
        candidates.append((site_locs, path, mapping))
    if not candidates:
        return None
    best_locs = min(row[0] for row in candidates)
    winners = [row for row in candidates if row[0] == best_locs]
    valence_term = MULT[acid_count] + "yl"
    polyol_parent = f"{stem}ane-{','.join(map(str, best_locs))}-{valence_term}"
    from .component_mapping import MappedComponent, decorate_component
    locant_maps = tuple(sorted({tuple(sorted((idx, mapping[idx]) for idx in path))
                                for _, path, mapping in winners}))
    representative = winners[0]
    _site_locs, path, mapping = representative
    polyol_component = MappedComponent(
        parent=polyol_parent,
        atom_locants=tuple((idx, mapping[idx]) for idx in path),
        atoms=frozenset(path),
        systematic_parent=True)
    polyol_name = decorate_component(mol, Chem, polyol_component)
    if not polyol_name:
        return None
    name = f"{polyol_name} {acid_multiplication.text}"

    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    if any(atom.GetAtomicNum() not in {6, 8, 16} for atom in iter_atoms(mol)
           if atom.GetAtomicNum() > 1):
        return None
    acid_union = set().union(*map(set, acid_atom_sets))
    if acid_union & framework or acid_union & bridges or framework & bridges:
        return None
    if acid_union | framework | bridges != set(heavy):
        return None
    if sum(map(len, acid_atom_sets)) + len(framework) + len(bridges) != len(heavy):
        return None
    roles = []
    for index, atom_ids in enumerate(acid_atom_sets):
        roles.append((f"sulfur-acid-anion:{index + 1}", atom_ids,
                      min(acid_maps[index])))
    polyol_locants = tuple((idx, mapping[idx]) for idx in path)
    roles.append(("polyol-organyl", frozenset(framework), polyol_locants))
    roles.extend((f"ester-bridge-oxygen:{index}", frozenset((bridge,)), ())
                 for index, bridge in enumerate(sorted(bridges), 1))
    return SulfurPolyolEsterDecision(
        name=name,
        method="functional-class",
        principal_centers=frozenset(site.center for site, _ in sites),
        atoms=heavy,
        rule="P-65.6.3.2.1/P-65.6.3.3.3.1/P-65.6.1/P-65.3.1",
        atom_locants=tuple(row for _label, _atoms, rows in roles for row in rows),
        component_roles=tuple(roles),
        equivalent_polyol_atom_locants=locant_maps,
        equivalent_acid_atom_locants=tuple(acid_maps),
        acid_component_proofs=tuple(row[0] for row in acid_proofs),
        acid_copy_to_original=tuple(row[1] for row in acid_proofs),
        anion_multiplication=acid_multiplication,
    )


_RULE = "P-65.6.3.2.1/P-65.6.3.3.1/P-65.6.1/P-65.3.1"


def _pending(reason: str, *, general_construction_permitted=False) -> SulfurEsterFrontier:
    return SulfurEsterFrontier("pending", True, None, (), _RULE, reason,
                               general_construction_permitted)


def sulfur_ester_frontier(mol, Chem) -> SulfurEsterFrontier:
    """Return a mapped neutral sulfonate/sulfinate ester or explicit pending.

    The supported complete family has one sulfur acid ester, carbon-only
    saturated-tree or plain-phenyl organyl arms, ordinary valence, and no
    other heteroatom sites, charges, or unsupported annotations. Carbon isotope
    and tetrahedral descriptors are composed component-locally.
    A recognized ester whose other atoms fall outside the bounded arm grammar
    is kept pending. Where a genuinely senior group is present, P-65.6.3.2.3
    also requires prefix-mode ester naming; that rule is not inferred for
    subordinate groups such as hydroxy.
    """
    site = _recognize(mol)
    if site is None:
        return SulfurEsterFrontier("unmatched", False, None, (), _RULE,
                                   "no single neutral R-S(O)x-O-R' site was proved")
    if mol is None or mol.GetNumHeavyAtoms() > 96 or mol.GetStereoGroups():
        return _pending("whole-graph resource or stereo-group annotation is outside the bounded producer")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return _pending("charged or radical sulfur ester requires a separate salt/ion method")
    if any(a.GetIsotope() and a.GetAtomicNum() != 6 for a in iter_atoms(mol)):
        return _pending("non-carbon isotope grammar is outside this sulfur-ester route")
    if any(int(a.GetChiralTag()) and a.GetAtomicNum() != 6 for a in iter_atoms(mol)):
        return _pending("non-carbon stereochemistry is outside this sulfur-ester route")
    if any(b.GetStereo() or b.GetBondDir() != Chem.BondDir.NONE for b in iter_bonds(mol)):
        return _pending("bond stereochemistry is outside this ester grammar")
    if not _ordinary_valence(mol, Chem, site):
        return SulfurEsterFrontier("unmatched", False, None, (), _RULE,
                                   "the sulfur/oxygen/carbon valences are outside the proved neutral ester pattern")

    acid_carbons = _carbon_component(mol, site.acid_carbon)
    bridge_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(site.bridge_oxygen).GetNeighbors()
                        if a.GetIdx() != site.center]
    if len(bridge_neighbors) != 1 or mol.GetAtomWithIdx(bridge_neighbors[0]).GetAtomicNum() != 6:
        return _pending("the hydroxylic organyl arm is not one carbon-rooted component")
    organyl_carbons = _carbon_component(mol, bridge_neighbors[0])
    if acid_carbons is None or organyl_carbons is None:
        return _pending("one organyl arm is outside the saturated carbon-tree/plain-phenyl grammar",
                        general_construction_permitted=True)
    if acid_carbons & organyl_carbons:
        return _pending("acid and hydroxylic carbon components are not disjoint")
    carbon_annotations = {
        a.GetIdx() for a in iter_atoms(mol)
        if a.GetAtomicNum() == 6 and (a.GetIsotope() or int(a.GetChiralTag()))
    }
    if carbon_annotations - (acid_carbons | organyl_carbons):
        return _pending("a carbon annotation is not owned by either carbon component")

    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    acid_oxygens = frozenset((*site.oxo_atoms, site.bridge_oxygen))
    acid_component = frozenset((*acid_carbons, site.center, *acid_oxygens))
    organyl_component = frozenset(organyl_carbons)
    bridge_component = frozenset((site.bridge_oxygen,))
    # The bridge oxygen is assigned to the functional-class linkage, not to
    # either disjoint atom component. Its own role remains explicit.
    if (acid_component - bridge_component) | organyl_component | bridge_component != heavy:
        return _pending("extra atoms or competing functionality are not owned by this monoester method",
                        general_construction_permitted=True)
    if (acid_component - bridge_component) & organyl_component:
        return _pending("acid and organyl ownership overlap")

    acid_mol, new_to_old = _acid_form_component(mol, Chem, site, acid_carbons)
    if acid_mol is None:
        return _pending("the mapped acid-form component could not be constructed safely")
    from .extended_inventory import selection_inventory
    from .complete_functional_parent import functional_parent_candidates
    from .parent_selection import parent_frontier

    inventory = selection_inventory(acid_mol)
    acid_kind = "sulfonic-acid" if site.oxidation == 2 else "sulfinic-acid"
    groups = tuple(g for g in inventory.groups if g.kind == acid_kind)
    if len(groups) != 1 or len(inventory.groups) != 1:
        return _pending("the acid-form component does not have one isolated source-class acid group")
    candidates = functional_parent_candidates(acid_mol, Chem, include_single=True,
                                              principal=groups)
    if not candidates or any(c.parent_selection is None for c in candidates):
        return _pending("the acid-parent candidate frontier contains an unavailable comparison")
    maximal = parent_frontier(candidates, lambda c: c.parent_selection)
    if not maximal:
        return _pending("the acid-parent comparison returned no maximal candidate")
    acid_spellings = {c.name for c in maximal}
    parent_scopes = {frozenset(c.parent_atoms) for c in maximal}
    if len(acid_spellings) != 1 or len(parent_scopes) != 1:
        return _pending("the source-ranked acid-parent frontier has unresolved spellings or scopes")
    # Decorate every source-maximal construction. A failed maximum must not
    # disappear in favor of a shorter, unannotated, or otherwise convenient
    # candidate.
    from .component_annotation_evidence import decorate_component_with_evidence
    acid_decorations = []
    for candidate in maximal:
        if candidate.construction is None:
            return _pending("a maximal acid parent lacks a mapped construction")
        decoration = decorate_component_with_evidence(
            acid_mol, Chem, candidate.construction,
            component_id="sulfur-ester-acid")
        if decoration is None or decoration.text != candidate.name:
            return _pending("a maximal acid parent lacks complete mapped annotation evidence")
        acid_decorations.append((candidate, decoration))
    if len({decoration.text for _, decoration in acid_decorations}) != 1:
        return _pending("the maximal mapped acid components have unresolved annotation spellings")
    acid_name = acid_decorations[0][1].text
    if site.oxidation == 2 and acid_name.endswith("sulfonic acid"):
        anion_name = acid_name[:-len("sulfonic acid")] + "sulfonate"
    elif site.oxidation == 1 and acid_name.endswith("sulfinic acid"):
        anion_name = acid_name[:-len("sulfinic acid")] + "sulfinate"
    else:
        return _pending("the acid suffix has no sourced anion conversion for this sulfur state")

    from .component_mapping import mapped_branch
    organyl_name, owned_organyl = mapped_branch(mol, Chem, bridge_neighbors[0], site.bridge_oxygen)
    if not organyl_name or frozenset(owned_organyl) != organyl_component:
        return _pending("the hydroxylic organyl name does not own its full original carbon arm")

    organyl_atom_locants = ()
    if carbon_annotations & organyl_component:
        organyl_component_map = _annotated_organyl_component(
            mol, Chem, bridge_neighbors[0], site.bridge_oxygen,
            organyl_component)
        if organyl_component_map is None:
            return _pending("the annotated organyl arm lacks a complete mapped component")
        organyl_decoration = decorate_component_with_evidence(
            mol, Chem, organyl_component_map,
            component_id="sulfur-ester-organyl")
        if organyl_decoration is None or organyl_decoration.text != organyl_name:
            return _pending("the annotated organyl spelling/evidence differs from the mapped branch writer")
        organyl_atom_locants = tuple(sorted(
            (idx, str(locant)) for idx, locant in organyl_component_map.atom_locants
            if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
        ))

    maps = []
    acid_original_records = []
    acid_copy_maps = []
    acid_original_stereo = []
    for candidate, decoration in acid_decorations:
        mapped = []
        for atom_idx, locant in candidate.atom_locants:
            if atom_idx not in new_to_old:
                return _pending("the acid parent map references an atom outside its component")
            original = new_to_old[atom_idx]
            if original == site.bridge_oxygen:
                continue
            mapped.append((original, str(locant)))
        maps.append(tuple(sorted(mapped)))
        if not _acid_stereo_matches_original(
                mol, acid_mol, new_to_old, candidate.construction,
                decoration.stereo_atoms, Chem):
            return _pending("acid-component CIP descriptors change under the mapped ester-to-acid operation")
        acid_copy_maps.append(tuple(sorted(new_to_old.items())))
        translated = _translate_acid_isotope_record(
            mol, decoration.isotope_record, new_to_old,
            omitted_atom=site.bridge_oxygen,
            component_id="sulfur-ester-acid",
            component_text=anion_name)
        if translated is None:
            return _pending("acid isotope evidence cannot be translated onto original atoms")
        acid_original_records.append(translated)
        for evidence in decoration.stereo_atoms:
            old = new_to_old.get(evidence.atom_id)
            if old is None:
                return _pending("acid stereo evidence references an unmapped copy atom")
            acid_original_stereo.append(SulfurEsterCIPEvidence(
                old, "sulfur-ester-acid", evidence.component_locant,
                evidence.descriptor))
    if not maps or any(not m for m in maps):
        return _pending("the acid parent has no complete locant map")
    # Retain all source-equivalent maps and choose one deterministic record.
    unique_maps = tuple(sorted(set(maps)))
    acid_atom_locants = min(unique_maps)
    principal_center = frozenset((site.center,))
    decision_values = dict(
        name=f"{organyl_name} {anion_name}",
        method="functional-class",
        principal_centers=principal_center,
        atoms=heavy,
        rule=_RULE,
        atom_locants=acid_atom_locants,
        component_roles=(
            ("sulfur-acid-anion", acid_component - bridge_component, acid_atom_locants),
            ("organyl", organyl_component, organyl_atom_locants),
            ("ester-bridge-oxygen", bridge_component, ()),
        ),
    )
    isotope_records = []
    original_stereo = list(acid_original_stereo)
    if carbon_annotations:
        organyl_proof = None
        if carbon_annotations & organyl_component:
            organyl_proof = organyl_decoration
            organyl_record = organyl_decoration.isotope_record
            if organyl_record.component_id != "sulfur-ester-organyl":
                return _pending("organyl isotope record has an unexpected component identity")
            isotope_records.append(organyl_record)
            original_stereo.extend(
                SulfurEsterCIPEvidence(
                    row.atom_id, "sulfur-ester-organyl",
                    row.component_locant, row.descriptor)
                for row in organyl_decoration.stereo_atoms
            )
        else:
            organyl_proof = None
        # One source-maximal acid map is sufficient for whole-graph coverage;
        # all tied source-equivalent maps and copy origins remain attached to
        # the decision for callers that need the full frontier.
        if acid_original_records:
            isotope_records.append(acid_original_records[0])
        from .isotope_component_coverage import validate_isotope_component_coverage
        coverage = validate_isotope_component_coverage(mol, isotope_records)
        if not coverage.complete:
            return _pending("original isotope coverage is incomplete: " + "; ".join(coverage.errors))
        if not _validate_original_cip_evidence(mol, original_stereo, Chem):
            return _pending("original-graph CIP evidence is incomplete or inconsistent")
        decision = SulfurEsterDecision(
            **{**decision_values, "rule": _RULE + "/P-82/P-9"},
            equivalent_atom_locant_maps=unique_maps,
            acid_component_proofs=tuple(decoration for _, decoration in acid_decorations),
            acid_copy_to_original=tuple(acid_copy_maps),
            organyl_component_proof=organyl_proof,
            isotope_component_records=tuple(isotope_records),
            isotope_coverage=coverage,
            original_cip_evidence=tuple(sorted(
                set(original_stereo),
                key=lambda row: (row.component_id, row.component_locant, row.atom_id))),
        )
    else:
        decision = MethodDecision(**decision_values)
    return SulfurEsterFrontier("complete", True, decision, unique_maps, _RULE,
                               "neutral monoester roles and source-ranked acid parent are complete")


def sulfur_ester_decision(mol, Chem) -> MethodDecision | None:
    """Convenience wrapper for consumers that accept only complete decisions."""
    frontier = sulfur_ester_frontier(mol, Chem)
    return frontier.decision if frontier.disposition == "complete" else None


def mapped_sulfonate_decision(mol, Chem):
    """Join an original sulfur-derived prefix to a fully mapped organyl arm."""
    site = _recognize(mol)
    if (site is None or site.oxidation != 2 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope() or a.HasQuery()
                   for a in iter_atoms(mol))
            or not _ordinary_valence(mol, Chem, site)):
        return None
    from .component_mapping import (
        _general_branch_stereo_is_cited, fully_covered, has_annotations, mapped_general_branch,
    )
    from .extended_inventory import selection_inventory
    from .naming_result import SelectionLimitation

    groups = selection_inventory(mol).principal
    if len(groups) != 1 or (groups[0].kind, groups[0].center) != ('sulfur-oxoacid-ester', site.center):
        return None
    oxygen = mol.GetAtomWithIdx(site.bridge_oxygen)
    organyl_root = next(a.GetIdx() for a in oxygen.GetNeighbors() if a.GetIdx() != site.center)
    acid, acid_atoms = mapped_general_branch(mol, Chem, site.center, site.bridge_oxygen)
    organyl, organyl_atoms = mapped_general_branch(mol, Chem, organyl_root, site.bridge_oxygen)
    bridge = frozenset({site.bridge_oxygen})
    if (not acid.endswith('sulfonyl') or not organyl or any(c.isspace() for c in acid + organyl)
            or acid_atoms & organyl_atoms or bridge & (acid_atoms | organyl_atoms)
            or not fully_covered(mol, bridge, acid_atoms | organyl_atoms)
            or has_annotations(mol, acid_atoms)
            or not _general_branch_stereo_is_cited(mol, Chem, organyl_atoms, organyl)):
        return None
    name = organyl + ' ' + acid.removesuffix('sulfonyl') + 'sulfonate'
    return MethodDecision(
        name, 'functional-class', frozenset({site.center}), acid_atoms | bridge | organyl_atoms, _RULE,
        component_roles=(('sulfur-acid', acid_atoms, ()), ('ester-oxygen', bridge, ()),
                         ('organyl', organyl_atoms, ())),
        preference_issues=(SelectionLimitation(
            'sulfonate-components', _RULE,
            'Both components are mapped on the original ester; preferred parent comparison remains unresolved.'),))


def mapped_sulfur_ester_anion(mol, Chem, site):
    """Name the S-bound acid side separately from the alcohol arm."""
    from .branch_rules import collect_subtree
    from .component_mapping import decorate_component
    from .formatting import attach_suffix
    from .mapped_ring_parents import records

    root, = site.organyl
    if any(int(a.GetChiralTag()) for a in iter_atoms(mol)) or any(
            int(b.GetStereo()) for b in iter_bonds(mol)):
        return None  # Capping the acid must not change an assigned configuration.
    branch = frozenset(collect_subtree(mol, root, site.center))
    if site.center in branch:
        return None
    if mol.GetAtomWithIdx(root).IsInRing():
        choices = records(mol, Chem, scope=branch, attachment=root, general_branches=True)
        if len(choices) != 1:
            return None
        row = choices[0]
        component = row.construction
        if component is None or row.added_h_locants:
            return None
        locant = dict(row.atom_locants)[root]
        position = () if row.parent == 'benzene' and not component.prefixes else (locant,)
        anion = attach_suffix(row.hydride, site.anion, position)
        if not anion:
            return None
        rendered = decorate_component(mol, Chem, replace(component, parent=anion))
        return (rendered, branch) if rendered else None
    bounded = _SulfurEster(site.center, root, site.esters[0][0], site.oxo, len(site.oxo))
    acid, _new_to_old = _acid_form_component(mol, Chem, bounded, branch)
    if acid is None:
        return None
    from .complete_functional_parent import functional_parent_frontier

    frontier = functional_parent_frontier(acid, Chem, include_single=True)
    ending = 'sulfonic acid' if len(site.oxo) == 2 else 'sulfinic acid'
    if frontier.status not in {'complete', 'tie'} or not frontier.candidates:
        return None
    name = frontier.candidates[0].name
    return (name.removesuffix(ending) + site.anion, branch) if name.endswith(ending) else None


def _recognize(mol) -> _SulfurEster | None:
    if mol is None:
        return None
    centers = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 16]
    if len(centers) != 1:
        return None
    sulfur = centers[0]
    if sulfur.IsInRing():
        return None
    oxo = tuple(sorted(a.GetIdx() for a in sulfur.GetNeighbors()
                       if a.GetAtomicNum() == 8 and a.GetDegree() == 1
                       and mol.GetBondBetweenAtoms(sulfur.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2))
    bridge = [a.GetIdx() for a in sulfur.GetNeighbors()
              if a.GetAtomicNum() == 8
              and mol.GetBondBetweenAtoms(sulfur.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
    carbon = [a.GetIdx() for a in sulfur.GetNeighbors()
              if a.GetAtomicNum() == 6
              and mol.GetBondBetweenAtoms(sulfur.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
    if (len(oxo) not in {1, 2} or len(bridge) != 1 or len(carbon) != 1
            or len(sulfur.GetNeighbors()) != len(oxo) + 2):
        return None
    oxygen = mol.GetAtomWithIdx(bridge[0])
    if oxygen.IsInRing() or oxygen.GetDegree() != 2:
        return None
    if not any(a.GetIdx() != sulfur.GetIdx() and a.GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(oxygen.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1
               for a in oxygen.GetNeighbors()):
        return None
    return _SulfurEster(sulfur.GetIdx(), carbon[0], bridge[0], oxo, len(oxo))


def _carbon_component(mol, root: int) -> frozenset[int] | None:
    """Admit an all-single acyclic carbon tree or one plain phenyl ring."""
    if mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
        return None
    atoms = {root}
    stack = [root]
    while stack:
        idx = stack.pop()
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetAtomicNum() != 6:
                continue
            bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
            if bond.GetBondTypeAsDouble() != 1 and not bond.GetIsAromatic():
                return None
            if neighbor.GetIdx() not in atoms:
                atoms.add(neighbor.GetIdx())
                stack.append(neighbor.GetIdx())
    rings = [set(r) & atoms for r in mol.GetRingInfo().AtomRings() if set(r) & atoms]
    bonds = [b for b in iter_bonds(mol) if b.GetBeginAtomIdx() in atoms
             and b.GetEndAtomIdx() in atoms]
    if rings:
        if (len(rings) != 1 or len(atoms) != 6 or len(rings[0]) != 6
                or len(bonds) != 6 or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
                or any(not b.GetIsAromatic() for b in bonds)):
            return None
        return frozenset(atoms)
    if (len(bonds) != len(atoms) - 1
            or any(b.GetBondTypeAsDouble() != 1 for b in bonds)):
        return None
    return frozenset(atoms)


def _acid_form_component(mol, Chem, site, acid_carbons):
    """Copy the acid component and convert only the O–C bridge into O–H."""
    atom_ids = sorted(set(acid_carbons) | {site.center, site.bridge_oxygen, *site.oxo_atoms})
    rw = Chem.RWMol()
    old_to_new = {}
    for idx in atom_ids:
        atom = Chem.Atom(mol.GetAtomWithIdx(idx))
        old_to_new[idx] = rw.AddAtom(atom)
    # A source ester bridge oxygen can carry NoImplicit=True even though the
    # acid-form copy must acquire its ordinary hydroxyl H. Reset only this
    # copied atom; never alter the caller's original graph.
    copied_bridge_oxygen = rw.GetAtomWithIdx(old_to_new[site.bridge_oxygen])
    copied_bridge_oxygen.SetAtomicNum(8)
    copied_bridge_oxygen.SetNumExplicitHs(0)
    copied_bridge_oxygen.SetNoImplicit(False)
    allowed = set(atom_ids)
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in allowed and right in allowed:
            rw.AddBond(old_to_new[left], old_to_new[right], bond.GetBondType())
    # The O–organyl bond is absent because its carbon component was excluded.
    acid = rw.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None, {}
    return acid, {new: old for old, new in old_to_new.items()}


def _annotated_organyl_component(mol, Chem, root, attachment, atoms):
    """Rebuild the mapped saturated prefix used by the annotated branch writer.

    The current shared branch API exposes text and ownership but not its
    construction. This local adapter mirrors its source-ranked candidate
    paths, then requires both the local decoration and public branch spelling
    to agree. Aromatic rings and embedded annotated branches remain outside
    this adapter.
    """
    from .annotated_substituent_selection import (
        _annotation_frontier, _scope_is_admitted,
    )
    from .component_mapping import (
        MappedComponent, decorate_component, hydrogen_bindings, mapped_branch,
    )
    from .constants import ROOTS

    scope = _scope_is_admitted(mol, root, attachment, atoms)
    if scope is None or set(scope) != set(atoms):
        return None
    carbon_atoms = frozenset(
        idx for idx in scope if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
    root_degree = sum(n.GetIdx() in carbon_atoms
                      for n in mol.GetAtomWithIdx(root).GetNeighbors())
    method = 1 if root_degree <= 1 else 2
    frontier = _annotation_frontier(mol, Chem, root, attachment, scope, method)
    if not frontier:
        return None
    rendered = []
    for candidate in frontier:
        path = candidate.path_atom_ids
        if len(path) not in ROOTS:
            return None
        mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        path_set = frozenset(path)
        prefixes, embedded = [], set()
        remaining = set(carbon_atoms - path_set)
        while remaining:
            seed = min(remaining)
            group, stack, boundary = set(), [seed], set()
            while stack:
                idx = stack.pop()
                if idx not in remaining:
                    continue
                remaining.remove(idx)
                group.add(idx)
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    other = neighbor.GetIdx()
                    if other in path_set:
                        boundary.add(other)
                    elif other in remaining:
                        stack.append(other)
            if len(boundary) != 1:
                return None
            host = next(iter(boundary))
            member = next((idx for idx in group
                           if any(n.GetIdx() == host
                                  for n in mol.GetAtomWithIdx(idx).GetNeighbors())), None)
            if member is None:
                return None
            text, owned = mapped_branch(mol, Chem, member, host)
            if not text or set(owned) != group or embedded & set(owned):
                return None
            embedded.update(owned)
            prefixes.append((mapping[host], text))
        if set(mapping) | embedded != set(scope):
            return None
        parent = (ROOTS[len(path)] + "yl" if method == 1 else
                  ROOTS[len(path)] + "an-" + mapping[root] + "-yl")
        bindings = hydrogen_bindings(mol, mapping)
        component = MappedComponent(
            parent, tuple(bindings.items()), frozenset(bindings),
            tuple(prefixes), frozenset(embedded), systematic_parent=True)
        text = decorate_component(mol, Chem, component)
        if not text:
            return None
        branch_text, branch_atoms = mapped_branch(mol, Chem, root, attachment)
        if branch_text != text or frozenset(branch_atoms) != frozenset(atoms):
            return None
        rendered.append(component)
    identities = {
        (component.parent, component.atom_locants, component.prefixes,
         component.embedded_atoms)
        for component in rendered
    }
    return rendered[0] if len(identities) == 1 else None


def _acid_stereo_matches_original(mol, acid_mol, new_to_old, construction,
                                  stereo_evidence, Chem):
    """Ensure the O–H-capped acid copy preserves original carbon CIP labels."""
    if construction is None:
        return False
    mapped_atoms = {
        new_to_old[idx] for idx in construction.atoms
        if idx in new_to_old and mol.GetAtomWithIdx(new_to_old[idx]).GetAtomicNum() == 6
    }
    if any(int(mol.GetAtomWithIdx(idx).GetChiralTag()) for idx in mapped_atoms):
        try:
            from rdkit.Chem import rdCIPLabeler
            source = Chem.Mol(mol)
            capped = Chem.Mol(acid_mol)
            rdCIPLabeler.AssignCIPLabels(source)
            rdCIPLabeler.AssignCIPLabels(capped)
        except (AttributeError, RuntimeError, ValueError):
            return False
        for original in mapped_atoms:
            if not int(mol.GetAtomWithIdx(original).GetChiralTag()):
                continue
            acid_idx = next((idx for idx, old in new_to_old.items()
                             if old == original), None)
            if acid_idx is None:
                return False
            source_atom = source.GetAtomWithIdx(original)
            acid_atom = capped.GetAtomWithIdx(acid_idx)
            source_cip = source_atom.GetProp("_CIPCode") if source_atom.HasProp("_CIPCode") else None
            acid_cip = acid_atom.GetProp("_CIPCode") if acid_atom.HasProp("_CIPCode") else None
            if source_cip is None or source_cip != acid_cip:
                return False
    expected = set()
    for evidence in stereo_evidence:
        if evidence.atom_id not in new_to_old:
            return False
        expected.add(new_to_old[evidence.atom_id])
    actual = {
        idx for idx in mapped_atoms
        if mol.GetAtomWithIdx(idx).HasProp("_CIPCode")
    }
    # RDKit properties may not be populated on the source molecule; compare
    # the label sets only when at least one mapped center was annotated.
    if any(int(mol.GetAtomWithIdx(idx).GetChiralTag()) for idx in mapped_atoms):
        source = Chem.Mol(mol)
        try:
            from rdkit.Chem import rdCIPLabeler
            rdCIPLabeler.AssignCIPLabels(source)
        except (AttributeError, RuntimeError, ValueError):
            return False
        actual = {idx for idx in mapped_atoms
                  if source.GetAtomWithIdx(idx).HasProp("_CIPCode")}
    return actual == expected


def _translate_acid_isotope_record(mol, record, new_to_old, *, omitted_atom,
                                   component_id, component_text):
    """Rebuild the acid copy's isotope ledger on original graph atom IDs."""
    from .isotope_component_coverage import component_record_from_atom_map

    try:
        copied_scope = tuple(record.atom_ids)
        old_scope = tuple(new_to_old[idx] for idx in copied_scope
                          if idx in new_to_old and new_to_old[idx] != omitted_atom)
        if len(old_scope) != len(set(old_scope)) or not old_scope:
            return None
        map_rows = []
        for idx, locant in record.atom_locants:
            if idx not in new_to_old:
                return None
            original = new_to_old[idx]
            if original == omitted_atom:
                continue
            # Preserve an existing locant or None from the mapped writer.
            # Never synthesize a locant for the ester bridge oxygen.
            map_rows.append((original, locant))
        claims = []
        printed = {}
        for claim in record.isotope_atoms:
            if claim.atom_id not in new_to_old:
                return None
            original = new_to_old[claim.atom_id]
            if original == omitted_atom or not 0 <= original < mol.GetNumAtoms():
                return None
            atom = mol.GetAtomWithIdx(original)
            if atom.GetSymbol() != claim.element or atom.GetIsotope() != claim.mass:
                return None
            claims.append(original)
            printed[original] = claim.printed_locant
        translated = component_record_from_atom_map(
            mol,
            component_id=component_id,
            component_text=component_text,
            atom_ids=old_scope,
            atom_locants=tuple(map_rows),
            descriptor_atom_ids=tuple(claims),
            printed_locants=printed,
        )
        expected = {(new_to_old[claim.atom_id], claim.element, claim.mass,
                     claim.component_locant, claim.printed_locant)
                    for claim in record.isotope_atoms}
        actual = {(claim.atom_id, claim.element, claim.mass,
                   claim.component_locant, claim.printed_locant)
                  for claim in translated.isotope_atoms}
        if expected != actual:
            return None
        return translated
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


def _validate_original_cip_evidence(mol, evidence, Chem):
    """Check complete carbon stereocenter evidence against the input graph."""
    try:
        from rdkit.Chem import rdCIPLabeler
        original = Chem.Mol(mol)
        rdCIPLabeler.AssignCIPLabels(original)
        claimed = {}
        for row in evidence:
            atom_id, locant, descriptor = (
                row.atom_id, row.component_locant, row.descriptor)
            if atom_id in claimed or not isinstance(locant, str) or not locant:
                return False
            atom = original.GetAtomWithIdx(atom_id)
            if (atom.GetAtomicNum() != 6 or not int(atom.GetChiralTag())
                    or not atom.HasProp("_CIPCode")
                    or atom.GetProp("_CIPCode") != descriptor):
                return False
            claimed[atom_id] = descriptor
        required = {atom.GetIdx() for atom in original.GetAtoms()
                    if atom.GetAtomicNum() == 6 and int(atom.GetChiralTag())}
        return set(claimed) == required
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
        return False


def _ordinary_valence(mol, Chem, site) -> bool:
    """Require the ordinary neutral S(IV)/S(VI), O(II), and C(IV) pattern."""
    try:
        sulfur = mol.GetAtomWithIdx(site.center)
        expected_s_valence = 6 if site.oxidation == 2 else 4
        if (sulfur.GetFormalCharge() != 0 or sulfur.GetNumRadicalElectrons()
                or sulfur.GetValence(Chem.ValenceType.EXPLICIT) != expected_s_valence
                or sulfur.GetTotalValence() != expected_s_valence
                or sulfur.GetTotalNumHs(includeNeighbors=True) != 0):
            return False
        for idx in (*site.oxo_atoms, site.bridge_oxygen):
            oxygen = mol.GetAtomWithIdx(idx)
            if (oxygen.GetAtomicNum() != 8 or oxygen.GetFormalCharge() != 0
                    or oxygen.GetNumRadicalElectrons()
                    or oxygen.GetValence(Chem.ValenceType.EXPLICIT) != 2
                    or oxygen.GetTotalValence() != 2
                    or oxygen.GetTotalNumHs(includeNeighbors=True) != 0):
                return False
        for atom in iter_atoms(mol):
            if atom.GetAtomicNum() == 6 and (
                    atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons()
                    or atom.GetTotalValence() != 4):
                return False
    except (AttributeError, RuntimeError, ValueError):
        return False
    return True
