"""Mapped functional-class construction for partially esterified polyols.

The initial operation is limited to one saturated, unbranched carbon-chain
alcohol component with multiple ester links and at least one residual OH.
"""

from dataclasses import dataclass

from .component_mapping import MappedComponent, decorate_component
from .component_annotation_evidence import MappedComponentDecoration
from .constants import MULT, ROOTS
from .formatting import prefix_alphabetization_key
from .isotope_component_coverage import IsotopeComponentRecord, IsotopeCoverageResult
from .naming_methods import MethodDecision, NamingMethodOutcome, NamingMethodStatus
from .acid_anion_multiplication import AnionMultiplication
from .acid_anion_ordering import AcidAnionOrderAssessment
from .naming_result import SelectionLimitation


@dataclass(frozen=True, slots=True)
class PartialPolyolEsterDecision(MethodDecision):
    """Functional-class ester decision with every equivalent polyol map."""

    equivalent_organyl_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()
    organyl_decoration: MappedComponentDecoration | None = None
    acid_decorations: tuple[MappedComponentDecoration, ...] = ()
    hydroxy_isotope_prefixes: tuple[object, ...] = ()
    isotope_components: tuple[IsotopeComponentRecord, ...] = ()
    isotope_coverage: IsotopeCoverageResult | None = None
    anion_multiplications: tuple[AnionMultiplication, ...] = ()
    citation_order_evidence: AcidAnionOrderAssessment | None = None
    hydroxy_prefix_citation_evidence: object | None = None


@dataclass(frozen=True, slots=True)
class HydroxyPrefixCitationEvidence:
    """Typed P-82 citation fact for residual hydroxy units."""

    status: str
    rule: str
    profiles: tuple[tuple[int, tuple[int, ...]], ...]
    citation_order: tuple[str, ...] | None
    reason: str


def _simple_acid_component(mol, Chem, part):
    """Prove an unmodified, saturated, unbranched carboxylic acid arm."""
    atoms = set(part.atoms)
    if not atoms or part.carbonyl_oxygen not in atoms:
        return False
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in atoms):
        return False
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           or mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
        return False
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag())
           and mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in atoms):
        return False
    if sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in atoms) != 1:
        return False
    # The carbonyl center is the unique carbon with a double bond to the
    # component's sole oxygen atom.
    centers = [i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
               and mol.GetBondBetweenAtoms(i, part.carbonyl_oxygen) is not None
               and mol.GetBondBetweenAtoms(i, part.carbonyl_oxygen).GetBondTypeAsDouble() == 2]
    if len(centers) != 1:
        return False
    center = centers[0]
    carbons = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    adjacency = {i: set() for i in carbons}
    for i in carbons:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j in carbons:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond.GetBondTypeAsDouble() != 1:
                    return False
                adjacency[i].add(j)
    if (not carbons or any(len(neighbors) > 2 for neighbors in adjacency.values())
            or sum(map(len, adjacency.values())) // 2 != len(carbons) - 1
            or len(adjacency[center]) > 1):
        return False
    seen = set()
    stack = [center]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != carbons:
        return False
    # The acid component must be exactly the carbon chain, its carbonyl O,
    # and the acyl center; no other terminal/characteristic atoms are admitted.
    for i in atoms - {part.carbonyl_oxygen}:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            if (neighbor.GetIdx() not in atoms and neighbor.GetAtomicNum() > 1
                    and not (i == center and neighbor.GetIdx() == part.bridge_oxygen)):
                return False
    return True


def _simple_polyol_parent(mol, framework, attachment_atoms, hydroxy_atoms,
                          *, include_substituent_isotopes=False):
    """Return the unique source-ordered path frontier and mapped OH sites."""
    if len(framework) < 2:
        return None
    adjacency = {i: set() for i in framework}
    for i in framework:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetIsAromatic() or atom.IsInRing():
            return None
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            if j in framework:
                if mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1:
                    return None
                adjacency[i].add(j)
    if (any(len(neighbors) > 2 for neighbors in adjacency.values())
            or sum(map(len, adjacency.values())) // 2 != len(framework) - 1):
        return None
    leaves = tuple(i for i, neighbors in adjacency.items() if len(neighbors) == 1)
    if len(leaves) != 2:
        return None
    paths = []
    for start, end in (leaves, tuple(reversed(leaves))):
        path = [start]
        while path[-1] != end:
            next_atoms = adjacency[path[-1]] - set(path)
            if len(next_atoms) != 1:
                return None
            path.append(next(iter(next_atoms)))
        if len(path) == len(framework):
            paths.append(tuple(path))
    if not paths:
        return None
    candidates = []
    from .parent_annotations import isotope_numbering_key, isotope_seniority, stereo_numbering
    for path in paths:
        mapping = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        free_locs = tuple(sorted(int(mapping[i]) for i in attachment_atoms))
        oh_locs = tuple(sorted(int(mapping[i]) for i in hydroxy_atoms))
        stereo_key = stereo_numbering(mol, mapping)
        if stereo_key is None:
            return None
        isotope_key = (isotope_numbering_key(
            mol, mapping, include_substituent_groups=True)
            if include_substituent_isotopes else isotope_seniority(mol, mapping))
        if isotope_key is None:
            return None
        # P-46.1(b) is already satisfied by the only full chain. Criterion (h)
        # assigns the lowest free-valence set before the hydroxy-substituent
        # criteria. Isotope seniority is then compared on the mapped parent
        # (P-82.5.2), followed by original-graph CIP descriptors.
        key = (free_locs, -len(oh_locs), oh_locs, isotope_key, stereo_key)
        candidates.append((key, path, mapping, free_locs, oh_locs))
    best = min(row[0] for row in candidates)
    winners = [row for row in candidates if row[0] == best]
    return tuple(winners)


def partial_polyol_ester_decision(mol, Chem, groups, atoms, centers):
    """Return a source-bounded single-polyol partial-ester construction."""
    if len(groups) < 2 or mol.GetStereoGroups():
        return None
    if (any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
            or atom.GetIsotope() and atom.GetAtomicNum() not in {1, 6, 8}
            or atom.GetAtomicNum() == 1 and atom.GetIsotope() not in {1, 2, 3}
            for atom in mol.GetAtoms())):
        return None
    if any(int(bond.GetStereo()) for bond in mol.GetBonds()):
        return None
    if any(bond.GetBondDir() != Chem.BondDir.NONE for bond in mol.GetBonds()):
        return None
    if any(atom.GetAtomicNum() > 1 and (
            atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
            or atom.GetTotalValence() != (4 if atom.GetAtomicNum() == 6 else 2)
            or atom.GetAtomicNum() not in {6, 8}) for atom in mol.GetAtoms()):
        return None
    from .acid_derivative_methods import _acid_component, _anion

    if any(getattr(group, "kind", None) != "ester" for group in groups):
        return None
    if len({group.center for group in groups}) != len(groups):
        return None
    bridges = tuple(group.ligand for group in groups)
    if len(set(bridges)) != len(bridges):
        return None

    parts = []
    ester_centers = set()
    for group in groups:
        part = _acid_component(mol, Chem, group.center, group.ligand)
        if part is None or not _simple_acid_component(mol, Chem, part):
            return None
        anion = _anion(part.name)
        if not anion:
            return None
        parts.append(part)
        ester_centers.add(group.center)

    acid_atoms = set()
    for part in parts:
        if acid_atoms & set(part.atoms):
            return None
        acid_atoms.update(part.atoms)
    heavy_atoms = frozenset(i for i in atoms
                            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    if not (acid_atoms | set(bridges)) <= set(heavy_atoms):
        return None
    alcohol_atoms = set(heavy_atoms) - acid_atoms
    if acid_atoms | alcohol_atoms != set(heavy_atoms) or not set(bridges) <= alcohol_atoms:
        return None

    framework = {i for i in alcohol_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    bridge_set = set(bridges)
    hydroxy_atoms = set()
    hydroxy_oxygen_atoms = set()
    hydroxy_oxygen_by_host = {}
    for idx in alcohol_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 8 or idx in bridge_set:
            continue
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                     if neighbor.GetAtomicNum() > 1]
        if (len(neighbors) != 1 or neighbors[0] not in framework
                or atom.GetTotalNumHs(includeNeighbors=True) != 1
                or mol.GetBondBetweenAtoms(idx, neighbors[0]).GetBondTypeAsDouble() != 1):
            return None
        if neighbors[0] in hydroxy_atoms:
            # Repeated same-carbon OH locants require a distinct source-proved
            # locant/multiplication grammar (e.g. duplicated locants).
            return None
        hydroxy_atoms.add(neighbors[0])
        hydroxy_oxygen_atoms.add(idx)
        hydroxy_oxygen_by_host[neighbors[0]] = idx
    if not hydroxy_atoms or not framework:
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 8} for i in alcohol_atoms):
        return None
    if {i for i in alcohol_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 8} != (
            bridge_set | hydroxy_oxygen_atoms):
        return None
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) and
           mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           for i in range(mol.GetNumAtoms())):
        return None

    # The shared Job219 helper owns isotope syntax for one separately named
    # residual hydroxy unit. Explicit isotope hydrogens must be attached to
    # that OH oxygen; no isotope-bearing O elsewhere is admitted here.
    explicit_hydrogens = {atom.GetIdx() for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() == 1}
    isotope_hydrogens = set()
    for hydrogen_idx in explicit_hydrogens:
        hydrogen = mol.GetAtomWithIdx(hydrogen_idx)
        neighbors = [neighbor.GetIdx() for neighbor in hydrogen.GetNeighbors()]
        if (hydrogen.GetIsotope() not in {1, 2, 3} or len(neighbors) != 1
                or neighbors[0] not in hydroxy_oxygen_atoms):
            return None
        isotope_hydrogens.add(hydrogen_idx)
    isotope_oxygens = {atom.GetIdx() for atom in mol.GetAtoms()
                       if atom.GetAtomicNum() == 8 and atom.GetIsotope()}
    if isotope_oxygens - hydroxy_oxygen_atoms:
        return None
    labelled_hydroxy_oxygens = {
        oxygen for oxygen in hydroxy_oxygen_atoms
        if mol.GetAtomWithIdx(oxygen).GetIsotope()
        or any(neighbor.GetIdx() in isotope_hydrogens
               for neighbor in mol.GetAtomWithIdx(oxygen).GetNeighbors())
    }
    attachment_by_bridge = {}
    attachment_atoms = []
    for group, bridge in zip(groups, bridges):
        if mol.GetAtomWithIdx(bridge).GetAtomicNum() != 8 or int(mol.GetAtomWithIdx(bridge).GetChiralTag()):
            return None
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(bridge).GetNeighbors()
                     if n.GetIdx() != group.center and n.GetAtomicNum() > 1]
        if len(neighbors) != 1 or neighbors[0] not in framework:
            return None
        host = neighbors[0]
        if mol.GetBondBetweenAtoms(bridge, host).GetBondTypeAsDouble() != 1:
            return None
        attachment_by_bridge[bridge] = host
        attachment_atoms.append(host)
    # Geminal ester sites need a separate source-closed multivalent locant
    # grammar; don't create a repeated free-valence locant by assumption.
    if len(set(attachment_atoms)) != len(attachment_atoms):
        return None

    # Every non-acid heavy atom must be in the admitted C/O polyol component;
    # the exact role union below catches hidden carbonyls or other functions.
    winners = _simple_polyol_parent(
        mol, framework, attachment_atoms, hydroxy_atoms,
        include_substituent_isotopes=bool(labelled_hydroxy_oxygens))
    if winners is None:
        return None
    multiplier = MULT.get(len(groups))
    root_stem = ROOTS.get(len(framework))
    if root_stem is None or multiplier is None:
        return None
    root = root_stem + "ane"
    valence_term = multiplier + "yl"

    anions = tuple(_anion(part.name) for part in parts)
    if any(not name for name in anions):
        return None
    same_acid = all(part.identity == parts[0].identity
                    and part.name == parts[0].name for part in parts[1:])
    if labelled_hydroxy_oxygens:
        # Keep OH-isotope citation outside unresolved acid-component rules.
        if (not same_acid
                or any(mol.GetAtomWithIdx(i).GetIsotope()
                       for part in parts for i in part.atoms)
                or any(int(mol.GetAtomWithIdx(i).GetChiralTag())
                       for part in parts for i in part.atoms)):
            return None
    for i, left in enumerate(parts):
        for right, right_anion in zip(parts[i + 1:], anions[i + 1:]):
            if left.identity != right.identity and left.name == right.name and anions[i] == right_anion:
                return None

    from .polyol_ester_methods import (
        _acid_anion_order_assessment, _multiply_acid_component,
        _ordinary_acid_components, _typed_acid_parent_decoration,
    )
    from .polyol_ester_isotopes import ester_oxygen_profile
    ordinary_parts = _ordinary_acid_components(mol, Chem, groups)
    if ordinary_parts is None:
        return None
    for left_index, left in enumerate(parts):
        for right_index in range(left_index + 1, len(parts)):
            right = parts[right_index]
            if ordinary_parts[left_index][:2] != ordinary_parts[right_index][:2]:
                continue
            if left.identity == right.identity and left.name == right.name:
                continue
            left_modified = any(mol.GetAtomWithIdx(i).GetIsotope()
                                for i in left.atoms)
            right_modified = any(mol.GetAtomWithIdx(i).GetIsotope()
                                 for i in right.atoms)
            # P-82.2.2.1 establishes modified-before-unmodified for the
            # otherwise identical pair. It does not settle two unlike
            # modified patterns, nor differing stereochemical components.
            if left_modified == right_modified:
                return None
    acid_decorations = []
    for index, (part, ordinary) in enumerate(zip(parts, ordinary_parts), 1):
        profile = ester_oxygen_profile(
            mol, part.carbonyl_oxygen, part.bridge_oxygen)
        if profile is None or profile.suffix_oxygen_isotopes:
            return None
        decoration = _typed_acid_parent_decoration(
            mol, Chem, part, profile, f"partial-polyol-acid:{index}")
        if decoration is None:
            return None
        carbon_labels = {idx for idx in part.atoms
                         if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
                         and mol.GetAtomWithIdx(idx).GetIsotope()}
        claims = {record.atom_id for record in decoration.isotope_record.isotope_atoms
                  if record.element == "C"}
        if claims != carbon_labels or any(
                record.element != "C" for record in decoration.isotope_record.isotope_atoms):
            return None
        acid_decorations.append(decoration)
    candidates = []
    from .polyol_hydroxy_isotopes import polyol_hydroxy_isotope_prefix
    for _key, path, mapping, free_locs, hydroxy_locs in winners:
        prefix_rows = []
        hydroxy_prefix_decorations = []
        for host in sorted(hydroxy_atoms, key=lambda idx: mapping[idx]):
            oxygen = hydroxy_oxygen_by_host[host]
            hydroxy_decoration = polyol_hydroxy_isotope_prefix(
                mol, Chem, host_atom_id=host, oxygen_atom_id=oxygen,
                host_locant=str(mapping[host]),
                component_id=f"partial-polyol-free-hydroxy:{mapping[host]}")
            if hydroxy_decoration is None:
                return None
            prefix_rows.append((str(mapping[host]), hydroxy_decoration.text))
            hydroxy_prefix_decorations.append(hydroxy_decoration)
        prefixes = tuple(prefix_rows)
        profile_by_text = {}
        for record in hydroxy_prefix_decorations:
            profile_by_text[record.text] = (
                mol.GetAtomWithIdx(record.oxygen_atom_id).GetIsotope(),
                tuple(sorted(mol.GetAtomWithIdx(idx).GetIsotope()
                             for idx in record.hydrogen_atom_ids
                             if mol.GetAtomWithIdx(idx).GetIsotope())))
        distinct_profiles = {profile for profile in profile_by_text.values()}
        natural_texts = [text for text, profile in profile_by_text.items()
                         if profile == (0, ())]
        modified_texts = [text for text, profile in profile_by_text.items()
                          if profile != (0, ())]
        if len(distinct_profiles) > 2 or len(modified_texts) > 1:
            prefix_citation_order = None
            prefix_citation_status = "pending-pin-order"
            prefix_citation_reason = "P-82.3 orders isotope symbols at one insertion site only"
        elif modified_texts and natural_texts:
            # P-82.2.2.1 expressly gives the modified otherwise-identical unit
            # priority over its unmodified counterpart. This typed order is
            # passed as component evidence; the formatter never infers it from
            # isotope punctuation or final name strings.
            prefix_citation_order = tuple(modified_texts + natural_texts)
            prefix_citation_status = "source-rule"
            prefix_citation_reason = "otherwise-identical modified unit precedes natural unit"
        else:
            prefix_citation_order = None
            prefix_citation_status = "source-rule"
            prefix_citation_reason = "no unlike modified/natural prefix pair"
        hydroxy_citation_evidence = HydroxyPrefixCitationEvidence(
            status=prefix_citation_status,
            rule="P-82.2.2.1",
            profiles=tuple(sorted(distinct_profiles)),
            citation_order=prefix_citation_order,
            reason=prefix_citation_reason,
        )
        parent = f"{root}-{','.join(map(str, free_locs))}-{valence_term}"
        mapped_parent = MappedComponent(
            parent=parent,
            atom_locants=tuple((idx, mapping[idx]) for idx in path),
            atoms=frozenset(path),
            prefixes=prefixes,
            systematic_parent=True,
            prefix_citation_order=prefix_citation_order,
        )
        from .component_annotation_evidence import decorate_component_with_evidence
        decoration = decorate_component_with_evidence(
            mol, Chem, mapped_parent, component_id="partial-polyol-organyl")
        organyl = decorate_component(mol, Chem, mapped_parent)
        if (not organyl or decoration is None or decoration.text != organyl
                or decoration.evidence_scope != mapped_parent.atoms):
            return None
        site_anions = tuple(sorted(
            (anions[index], int(mapping[attachment_atoms[index]]), index)
            for index in range(len(parts))))
        anion_multiplications = []
        if same_acid:
            multiplication = _multiply_acid_component(
                mol, parts[0], anions[0], len(parts), acid_decorations[0])
            if multiplication is None:
                return None
            anion_multiplications.append(multiplication)
            acid_citation = multiplication.text
        else:
            citation_groups = {}
            for anion, locant, index in site_anions:
                citation_groups.setdefault(anion, []).append((locant, index))
            citations = []
            ordinary_anions = tuple(row[2] for row in ordinary_parts)
            group_keys = {}
            for anion, entries in citation_groups.items():
                indexes = [index for _locant, index in entries]
                ordinary_names = {ordinary_anions[index] for index in indexes}
                if len(ordinary_names) != 1:
                    return None
                ordinary_name = next(iter(ordinary_names))
                modified = {any(mol.GetAtomWithIdx(i).GetIsotope()
                                for i in parts[index].atoms)
                            for index in indexes}
                if len(modified) > 1:
                    return None
                group_keys[anion] = (prefix_alphabetization_key(ordinary_name),
                                     0 if next(iter(modified), False) else 1)
            # P-14.5 order uses ordinary component words. P-82.2.2.1 gives
            # the modified member priority only for otherwise identical
            # modified/unmodified anion units.
            for anion in sorted(citation_groups, key=lambda value: group_keys[value]):
                entries = citation_groups[anion]
                site_locs = sorted(locant for locant, _index in entries)
                count = len(site_locs)
                if count > 1:
                    component_index = entries[0][1]
                    item_part = parts[component_index]
                    multiplication = _multiply_acid_component(
                        mol, item_part, anion, count, acid_decorations[component_index])
                    if multiplication is None:
                        return None
                    anion_multiplications.append(multiplication)
                    unit = multiplication.text
                else:
                    unit = anion
                citations.append(",".join(map(str, site_locs)) + "-" + unit)
            acid_citation = " ".join(citations)
        candidates.append((organyl, acid_citation, path, mapping,
                           free_locs, hydroxy_locs, site_anions, decoration,
                           tuple(hydroxy_prefix_decorations)))
    if not candidates:
        return None

    # The principal organyl numbering was compared before acid citation.
    # When unlike anions remain attached to different sites under tied
    # equivalent organyl mappings, source text does not resolve this route's
    # map in the bounded method, so decline instead of using atom order.
    textual = {(row[0], row[1]) for row in candidates}
    if not same_acid and len(textual) != 1:
        return None
    if len(textual) != 1:
        return None
    organyl, acid_citation = next(iter(textual))
    selected_rows = [row for row in candidates
                     if (row[0], row[1]) == (organyl, acid_citation)]
    atom_maps = tuple(sorted({
        tuple(sorted((int(atom), str(locant)) for atom, locant in row[3].items()))
        for row in candidates
    }))
    representative = selected_rows[0]
    path, mapping, free_locs, hydroxy_locs = (
        representative[2], representative[3], representative[4], representative[5])
    name = f"{organyl} {acid_citation}"
    if not name:
        return None
    organyl_decoration = representative[7]
    hydroxy_prefix_decorations = representative[8]
    isotope_components = tuple(
        [organyl_decoration.isotope_record] if organyl_decoration else []
    ) + tuple(decoration.isotope_record for decoration in acid_decorations) + tuple(
        decoration.component_record for decoration in hydroxy_prefix_decorations
        if decoration.component_record is not None)
    isotope_coverage = None
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        from .isotope_component_coverage import validate_isotope_component_coverage
        isotope_coverage = validate_isotope_component_coverage(mol, isotope_components)
        if not isotope_coverage.complete:
            return None
    expected_stereo_atoms = {
        atom.GetIdx() for atom in mol.GetAtoms()
        if int(atom.GetChiralTag())
    }
    claimed_stereo_atoms = {
        evidence.atom_id
        for decoration in (tuple(acid_decorations) + (organyl_decoration,))
        if decoration is not None
        for evidence in decoration.stereo_atoms
    }
    if expected_stereo_atoms != claimed_stereo_atoms:
        return None
    alcohol_locants = tuple((idx, mapping[idx]) for idx in path)
    roles = tuple((f"acid:{index + 1}", part.atoms, part.atom_locants)
                  for index, part in enumerate(parts))
    roles += (("alcohol", frozenset(alcohol_atoms), alcohol_locants),)
    if set().union(*(set(role_atoms) for _label, role_atoms, _locants in roles)) != set(heavy_atoms):
        return None
    if sum(len(role_atoms) for _label, role_atoms, _locants in roles) != len(heavy_atoms):
        return None
    locants = tuple(loc for part in parts for loc in part.atom_locants) + alcohol_locants
    citation_order_evidence = _acid_anion_order_assessment(mol, parts, ordinary_parts)
    preference_issues = []
    if citation_order_evidence.status != 'source_closed':
        preference_issues.append(SelectionLimitation(
            'partial-polyol-ester-method', '/'.join(citation_order_evidence.rules),
            citation_order_evidence.reason))
    if hydroxy_citation_evidence.status != 'source-rule':
        preference_issues.append(SelectionLimitation(
            'partial-polyol-ester-method', hydroxy_citation_evidence.rule,
            hydroxy_citation_evidence.reason))
    rule = ("P-65.6.3.2.1/P-65.6.3.3.3.1" if same_acid
            else "P-65.6.3.2.1/P-65.6.3.3.3.2")
    return PartialPolyolEsterDecision(
        name=name,
        method="functional-class",
        principal_centers=frozenset(centers),
        atoms=heavy_atoms,
        rule=rule,
        atom_locants=locants,
        component_roles=roles,
        equivalent_organyl_atom_locants=atom_maps,
        organyl_decoration=organyl_decoration,
        acid_decorations=tuple(acid_decorations),
        hydroxy_isotope_prefixes=hydroxy_prefix_decorations,
        isotope_components=isotope_components,
        isotope_coverage=isotope_coverage,
        anion_multiplications=tuple(anion_multiplications),
        citation_order_evidence=citation_order_evidence,
        hydroxy_prefix_citation_evidence=hydroxy_citation_evidence,
        preference_issues=tuple(preference_issues),
    )


def partial_polyol_ester_frontier(mol, Chem):
    """Return complete/pending/outside status for the residual-polyol method.

    Recognition is deliberately broader than the renderer: a complete set of
    principal esters sharing one cut-mapped alcoholic component with a free OH
    is method-eligible even when a ring, branch, label, or unresolved mapping
    prevents this bounded name constructor from completing.
    """
    from .extended_inventory import selection_inventory

    outside = lambda reason: NamingMethodOutcome(
        NamingMethodStatus.OUTSIDE_SCOPE, rule="P-65.6.3.3",
        method="functional-class", reason=reason)
    pending = lambda reason, centers=(): NamingMethodOutcome(
        NamingMethodStatus.PENDING, rule="P-65.6.3.3",
        method="functional-class", reason=reason,
        evidence=(("principal_ester_centers", tuple(sorted(centers))),))

    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol, asMols=False)) != 1):
        return outside("no-single-connected-polyol-ester-graph")
    inventory = selection_inventory(mol)
    if not inventory.supported:
        return outside("principal-inventory-not-proved")
    esters = tuple(group for group in inventory.groups if group.kind == "ester")
    principal = tuple(inventory.principal)
    centers = frozenset(group.center for group in esters)
    if (len(esters) < 2 or len(principal) != len(esters)
            or any(group.kind != "ester" for group in principal)
            or frozenset(group.center for group in principal) != centers):
        return outside("not-a-complete-principal-ester-set")

    # Cut every acid-side C-O ester bond. All bridge oxygens must map to one
    # alcoholic residue, and at least one original free OH group must map to
    # that same residue. An OH on an acid component alone does not qualify.
    cut = Chem.RWMol(mol)
    bridges = set()
    for group in esters:
        if (group.ligand < 0 or group.ligand in bridges
                or mol.GetBondBetweenAtoms(group.center, group.ligand) is None):
            return outside("ester-component-cuts-not-unique")
        bridges.add(group.ligand)
        cut.RemoveBond(group.center, group.ligand)
    fragmented = cut.GetMol()
    try:
        Chem.SanitizeMol(fragmented)
        mappings = []
        Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True,
                         fragsMolAtomMapping=mappings)
    except (ValueError, RuntimeError):
        return pending("ester-cut-component-mapping-failed", centers)
    owner = {atom: fragment for fragment, mapping in enumerate(mappings)
             for atom in mapping}
    bridge_owners = {owner.get(bridge) for bridge in bridges}
    if len(bridge_owners) != 1 or None in bridge_owners:
        return outside("ester-links-do-not-share-one-alcoholic-component")
    alcohol_component = next(iter(bridge_owners))
    if any(owner.get(group.center) == alcohol_component for group in esters):
        # An intramolecular/cyclic ester cut remains connected through the
        # original skeleton and is handled by the lactone/cyclic ester rules.
        return outside("cyclic-ester-not-a-disjoint-single-polyol-component")
    alcohol_group_present = False
    for group in inventory.groups:
        if group.kind != "alcohol" or group.center < 0:
            continue
        oxygen_sites = [idx for idx in group.atoms
                        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8
                        and idx not in bridges]
        if (owner.get(group.center) == alcohol_component
                and any(owner.get(idx) == alcohol_component for idx in oxygen_sites)):
            alcohol_group_present = True
            break
    if not alcohol_group_present:
        return outside("no-residual-oh-on-mapped-polyol-component")

    atoms = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1)
    decision = partial_polyol_ester_decision(mol, Chem, esters, atoms, centers)
    if decision is None:
        return pending("recognized-polyol-ester-rendering-or-comparison-blocked", centers)
    return NamingMethodOutcome(
        NamingMethodStatus.COMPLETE, decision,
        rule=decision.rule, method=decision.method, reason="rendered",
        evidence=(("principal_ester_centers", tuple(sorted(centers))),
                  ("hydroxy_prefix_citation", decision.hydroxy_prefix_citation_evidence)))
