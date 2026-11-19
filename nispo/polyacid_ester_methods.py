"""Mapped functional-class names for fully esterified polybasic acids."""

from collections import defaultdict
from dataclasses import replace

from .molecule_iteration import iter_atoms
from .component_mapping import mapped_branch
from .formatting import enclose_name_unit, format_prefixes, prefix_alphabetization_key
from .naming_methods import MethodDecision


def _stereo_signature(mol, Chem, *, atoms=None):
    from .stereochemistry import assigned_atom_stereochemistry, _indexed_bond_stereochemistry

    atom_set = None if atoms is None else set(atoms)
    atom_stereo = tuple((idx, descriptor) for idx, descriptor in
                        assigned_atom_stereochemistry(mol, Chem, atom_order="idx", include_pseudo=True)
                        if atom_set is None or idx in atom_set)
    bond_stereo = tuple((a, b, descriptor) for a, b, descriptor in
                        _indexed_bond_stereochemistry(mol, Chem)
                        if atom_set is None or {a, b} <= atom_set)
    return atom_stereo, bond_stereo


def _same_stereochemistry_after_remap(original, Chem, capped, capped_to_original, original_core):
    original_atoms, original_bonds = _stereo_signature(original, Chem, atoms=original_core)
    mapped_atoms, mapped_bonds = _stereo_signature(capped, Chem)
    back = dict(capped_to_original)
    mapped_atoms = tuple(sorted((back[idx], descriptor) for idx, descriptor in mapped_atoms
                                if idx in back and back[idx] in original_core))
    mapped_bonds = tuple(sorted((back[a], back[b], descriptor) for a, b, descriptor in mapped_bonds
                                if a in back and b in back and {back[a], back[b]} <= original_core))
    return original_atoms == mapped_atoms and original_bonds == mapped_bonds


def _swap_two_organyl_arms(mol, Chem, left, right):
    """Whether exchanging two original alcohol-side arms preserves the graph."""
    bridge_left, root_left = left["bridge"], left["root"]
    bridge_right, root_right = right["bridge"], right["root"]
    if bridge_left == bridge_right or root_left == root_right:
        return False
    rw = Chem.RWMol(mol)
    if (rw.GetBondBetweenAtoms(bridge_left, root_left) is None or
            rw.GetBondBetweenAtoms(bridge_right, root_right) is None):
        return False
    rw.RemoveBond(bridge_left, root_left)
    rw.RemoveBond(bridge_right, root_right)
    rw.AddBond(bridge_left, root_right, Chem.BondType.SINGLE)
    rw.AddBond(bridge_right, root_left, Chem.BondType.SINGLE)
    swapped = rw.GetMol()
    try:
        Chem.SanitizeMol(swapped)
    except (ValueError, RuntimeError):
        return False
    if _stereo_signature(mol, Chem) != _stereo_signature(swapped, Chem):
        return False
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == \
        Chem.MolToSmiles(swapped, canonical=True, isomericSmiles=True)


def _organyl_identity(mol, Chem, atoms, bridge, root):
    """Canonical rooted graph key retaining the ester oxygen and stereotags."""
    keep = set(atoms) | {bridge}
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(root).SetAtomMapNum(1)
    for idx in reversed(range(mol.GetNumAtoms())):
        if idx not in keep:
            rw.RemoveAtom(idx)
    bridge_new = bridge - sum(i < bridge for i in range(mol.GetNumAtoms()) if i not in keep)
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(2)
    dummy_idx = rw.AddAtom(dummy)
    rw.AddBond(bridge_new, dummy_idx, Chem.BondType.SINGLE)
    fragment = rw.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except (ValueError, RuntimeError):
        return ""
    return Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)


def _acid_site_locant(candidate, capped_index, capped_host_index=None):
    locants = dict(candidate.atom_locants)
    value = locants.get(capped_index)
    if value is None:
        return ""
    text = str(value)
    # For external carboxyl suffixes, the functional-parent renderer gives
    # the carbonyl a temporary C-locant. Ester-site locants belong to the
    # actual skeletal host, whose locant is independently mapped here.
    if text.startswith("C") and text[1:].isdigit():
        host = str(locants.get(capped_host_index, ""))
        return host if host.isdigit() else ""
    return text if text.isdigit() else ""


def _preferred_parent_mappings(candidates):
    """Retain tied numberings, but do not guess among distinct acid parents."""
    from .parent_selection import compare_parents

    by_parent = defaultdict(list)
    for candidate in candidates:
        identity = (frozenset(candidate.parent_atoms), candidate.principal_centers)
        by_parent[identity].append(candidate)
    parent_rows = list(by_parent.values())
    if len(parent_rows) > 1:
        keep = set(range(len(parent_rows)))
        for i, left_rows in enumerate(parent_rows):
            for j in range(i + 1, len(parent_rows)):
                right_rows = parent_rows[j]
                comparison = compare_parents(left_rows[0].parent_selection,
                                             right_rows[0].parent_selection)
                if comparison.preference is None:
                    return None
                if comparison.preference < 0:
                    keep.discard(j)
                elif comparison.preference > 0:
                    keep.discard(i)
        if len(keep) != 1:
            return None
        parent_rows = [parent_rows[next(iter(keep))]]
    mappings = parent_rows[0]
    # Prefer a mapping that has better proved P-44/P-45 parent facts before
    # considering the ester-site locants. Unknown comparisons stay tied for
    # the explicit numbering-key stage below.
    keep = set(range(len(mappings)))
    for i, left in enumerate(mappings):
        for j in range(i + 1, len(mappings)):
            comparison = compare_parents(left.parent_selection,
                                         mappings[j].parent_selection)
            if comparison.preference == -1:
                keep.discard(j)
            elif comparison.preference == 1:
                keep.discard(i)
    return [mappings[i] for i in sorted(keep)] or mappings


def _parent_numbering_key(candidate):
    """Source-ordered numbering criteria available on a capped acid parent."""
    # FunctionalParent.numbering_key's third field is the suffix-locant
    # sequence. Compare it as a set (P-14.3.5), then parent prefix locants
    # (P-14.4(f)); isotope and stereo annotation locants follow P-14.4(i)-(j).
    facts = candidate.parent_selection
    return (facts.suffix_locants, facts.prefix_locants,
            facts.prefix_citation_locants, facts.isotope_key,
            facts.stereo_numbering_key)


def _organyl_numbering_key(candidate, arms, ordered_species, old_to_capped):
    locants = dict(candidate.atom_locants)
    species_keys = [(item["identity"], item["name"]) for item in ordered_species]
    result = []
    for key in species_keys:
        sites = []
        for arm in arms:
            if (arm["identity"], arm["name"]) != key:
                continue
            center = old_to_capped.get(arm["group"].center)
            host = old_to_capped.get(arm["group"].owner)
            value = locants.get(center)
            if value is None:
                return None
            text = str(value)
            if text.startswith("C") and text[1:].isdigit():
                text = str(locants.get(host, ""))
            if not text.isdigit():
                return None
            sites.append(int(text))
        result.append(tuple(sorted(sites)))
    return tuple(result)


def _organyl_key(item, mol):
    alpha = prefix_alphabetization_key(item["name"])
    modified = any(mol.GetAtomWithIdx(i).GetIsotope() for i in item["atoms"])
    return alpha[0], alpha[1], 0 if modified else 1, alpha[2], item["name"]


def polyacid_ester_decision(mol, Chem):
    """Name a complete ester set on one polybasic carboxylic-acid component.

    The acid parent is selected once on a graph in which all ester bridge
    oxygens are retained as OH caps. Alcohol-derived components are separately
    mapped on the original graph, then cited before the acid-derived anion.
    Partial esters and salts decline under P-65.6.3.3.5, whose PIN method is
    substitutive on the anion.
    """
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                   for a in iter_atoms(mol)):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    groups = inventory.principal
    if len(groups) < 2 or any(group.kind != "ester" for group in groups):
        return None
    if sum(group.kind == "ester" for group in inventory.groups) != len(groups):
        return None
    if any(group.kind in {"acid", "carboxylate", "sulfonic-acid", "sulfonate"}
           for group in inventory.groups):
        return None
    # The multibridge O isotope placement is not established by P-82.6.4.
    for atom in iter_atoms(mol):
        if atom.GetIsotope() and atom.GetAtomicNum() != 6:
            return None

    from .complete_functional_parent import functional_parent_candidates
    from .acid_derivative_methods import _anion

    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    centers = frozenset(group.center for group in groups)
    arms = []
    bridges = set()
    for group in groups:
        bridge = group.ligand
        if bridge in bridges or mol.GetAtomWithIdx(bridge).GetAtomicNum() != 8:
            return None
        bridges.add(bridge)
        bridge_atom = mol.GetAtomWithIdx(bridge)
        if bridge_atom.GetIsotope() or int(bridge_atom.GetChiralTag()):
            return None
        carbonyl_oxygens = [n.GetIdx() for n in mol.GetAtomWithIdx(group.center).GetNeighbors()
                            if n.GetAtomicNum() == 8 and
                            mol.GetBondBetweenAtoms(group.center, n.GetIdx()).GetBondTypeAsDouble() == 2]
        if len(carbonyl_oxygens) != 1 or mol.GetAtomWithIdx(carbonyl_oxygens[0]).GetIsotope():
            return None
        roots = [n.GetIdx() for n in bridge_atom.GetNeighbors() if n.GetIdx() != group.center]
        if len(roots) != 1 or mol.GetAtomWithIdx(roots[0]).GetAtomicNum() != 6:
            return None
        name, owned = mapped_branch(mol, Chem, roots[0], bridge)
        if not name or not owned or group.center in owned or centers & owned or bridges & owned:
            return None
        if mol.GetBondBetweenAtoms(bridge, roots[0]).GetBondTypeAsDouble() != 1:
            return None
        arms.append({"group": group, "bridge": bridge, "root": roots[0],
                     "name": name, "atoms": frozenset(owned)})
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            if left["atoms"] & right["atoms"]:
                return None
    arm_atoms = set().union(*(set(item["atoms"]) for item in arms))
    if not arm_atoms or bridges & arm_atoms or not arm_atoms | bridges <= heavy:
        return None
    acid_core = set(heavy) - arm_atoms
    if bridges - acid_core or centers - acid_core:
        return None
    acid_seen, stack = set(), [next(iter(acid_core))]
    while stack:
        idx = stack.pop()
        if idx in acid_seen:
            continue
        acid_seen.add(idx)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in acid_core and n.GetIdx() not in acid_seen)
    if acid_seen != acid_core:
        return None

    # Delete all monovalent alcohol-side components simultaneously. Each
    # original bridge O remains on its mapped acid carbonyl and is thereby an
    # ordinary hydroxy cap for parent selection.
    rw = Chem.RWMol(mol)
    for idx in sorted(arm_atoms, reverse=True):
        rw.RemoveAtom(idx)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    removed = sorted(arm_atoms)
    old_to_capped = {i: i - sum(j < i for j in removed)
                     for i in range(mol.GetNumAtoms()) if i not in arm_atoms}
    capped_to_original = tuple(sorted((new, old) for old, new in old_to_capped.items()))
    from .extended_inventory import selection_inventory as capped_inventory
    acid_inventory = capped_inventory(capped)
    acid_groups = acid_inventory.principal
    if (len(acid_groups) != len(groups) or any(g.kind != "acid" for g in acid_groups) or
            {old_to_capped[g.center] for g in groups} != {g.center for g in acid_groups}):
        return None
    candidates = functional_parent_candidates(capped, Chem, include_single=True,
                                              principal=acid_groups)
    # The older functional-parent constructor leaves annotation facts absent.
    # Prove them from explicit graph/mapping data; rendered strings are not
    # evidence for either isotope or stereo numbering (P-82.5/P-14.4).
    from .parent_selection import graph_parent_facts
    mapped_candidates = []
    for option in candidates:
        facts = graph_parent_facts(
            capped, option.parent_atoms, dict(option.atom_locants),
            principal_class=acid_groups[0].seniority,
            suffix_centers=option.principal_centers,
            suffix_locants=option.numbering_key[2], prefixes=option.prefixes,
            kind='ring' if all(capped.GetAtomWithIdx(i).IsInRing()
                               for i in option.parent_atoms) else 'chain',
            name=option.name, allow_annotations=True)
        if facts is not None and all(value is not None for value in (
                facts.suffix_locants, facts.prefix_locants,
                facts.prefix_citation_locants, facts.isotope_key,
                facts.stereo_numbering_key)):
            mapped_candidates.append(replace(option, parent_selection=facts))
    candidates = tuple(mapped_candidates)
    if not candidates:
        return None
    candidate = candidates[0]
    acid_name = _anion(candidate.name)
    if not acid_name:
        return None
    if not _same_stereochemistry_after_remap(mol, Chem, capped, capped_to_original, acid_core):
        return None

    for item in arms:
        item["identity"] = _organyl_identity(mol, Chem, item["atoms"], item["bridge"], item["root"])
        if not item["identity"]:
            return None
    species = {}
    for item in arms:
        key = (item["identity"], item["name"])
        species.setdefault(key, {"name": item["name"], "identity": item["identity"],
                                 "items": [], "atoms": set()})
        species[key]["items"].append(item)
        species[key]["atoms"].update(item["atoms"])
    # Distinct graph components may not collapse to the same written group.
    species_rows = list(species.values())
    for i, left in enumerate(species_rows):
        for right in species_rows[i + 1:]:
            if left["name"] == right["name"]:
                return None

    # Decide locant necessity structurally. A distinct organyl assignment
    # needs locants if exchanging it with another unlike component creates a
    # different isotope/stereo-aware graph.
    needs_locant = set()
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            if (left["identity"], left["name"]) == (right["identity"], right["name"]):
                continue
            if not _swap_two_organyl_arms(mol, Chem, left, right):
                needs_locant.add((left["identity"], left["name"]))
                needs_locant.add((right["identity"], right["name"]))

    # P-82.5 follows the numbering of the isotopically unmodified compound.
    # The relative priority of functional-class organyl word locants and
    # acid-parent isotope/stereo locants is not established by this producer.
    # Do not turn that missing rule into a string or arbitrary ordering.
    from .component_mapping import has_annotations
    if needs_locant and has_annotations(mol, acid_core - bridges):
        return None

    ordered = sorted(species_rows, key=lambda item: _organyl_key(item, mol))
    candidate_options = _preferred_parent_mappings(candidates)
    if not candidate_options:
        return None
    # P-44/P-45 parent selection is resolved first. For equivalent numberings
    # of that same parent, use the P-14.4/P-45.2 tie order of unlike organyl
    # groups, rather than whichever capped-graph orientation was enumerated
    # first (which can depend on input atom order).
    candidate = min(
        candidate_options,
        key=lambda option: (
            _parent_numbering_key(option),
            _organyl_numbering_key(option, arms, ordered, old_to_capped)
            if _organyl_numbering_key(option, arms, ordered, old_to_capped) is not None
            else ((10**9,),),
        ),
    )
    acid_name = _anion(candidate.name)
    if not acid_name:
        return None
    site_to_locant = {}
    for group in groups:
        capped_center = old_to_capped[group.center]
        capped_host = old_to_capped.get(group.owner)
        site_to_locant[group.center] = _acid_site_locant(candidate, capped_center, capped_host)
        if not site_to_locant[group.center]:
            return None
    for arm in arms:
        arm["locant"] = site_to_locant[arm["group"].center]

    alcohol_words = []
    for item in ordered:
        count = len(item["items"])
        unit = (item["name"] if count == 1 else
                format_prefixes([("", item["name"])] * count, omit_locants=True))
        if count > 1:
            from .organyl_prefix_components import multiplied_organyl_prefix
            arm = item["items"][0]
            typed = multiplied_organyl_prefix(mol, arm["root"], arm["atoms"], item["name"], count)
            if typed is not None:
                unit = typed
        if not unit:
            return None
        key = (item["identity"], item["name"])
        if key in needs_locant:
            locants = sorted((arm["locant"] for arm in item["items"]), key=lambda value: (int(value), value))
            if unit[0].isdigit():
                unit = enclose_name_unit(unit)
            unit = ",".join(locants) + "-" + unit
        alcohol_words.append(unit)

    original_atom_locants = []
    for capped_idx, locant in candidate.atom_locants:
        old_idx = dict(capped_to_original).get(capped_idx)
        if old_idx is not None and old_idx not in bridges and old_idx in acid_core:
            original_atom_locants.append((old_idx, locant))
    acid_named_capped = (set(candidate.construction.atoms) | set(candidate.construction.embedded_atoms)
                         if candidate.construction is not None else set())
    if not acid_named_capped:
        return None
    acid_named_atoms = frozenset(dict(capped_to_original)[idx] for idx in acid_named_capped
                                 if idx in dict(capped_to_original) and
                                 dict(capped_to_original)[idx] in acid_core)
    acid_role_atoms = frozenset(acid_core - bridges)
    if not acid_role_atoms <= acid_named_atoms:
        return None
    if acid_role_atoms & arm_atoms or acid_role_atoms & bridges:
        return None
    roles = [("acid", acid_role_atoms, tuple(original_atom_locants))]
    for number, item in enumerate(arms, 1):
        roles.append((f"alcohol:{number}", frozenset(set(item["atoms"]) | {item["bridge"]}), ()))
    if set().union(*(set(role[1]) for role in roles)) != set(heavy):
        return None
    mapped = tuple(original_atom_locants)
    return MethodDecision(" ".join(alcohol_words + [acid_name]), "functional-class", centers,
                          heavy, "P-65.6.3.2.1/P-65.6.3.3.2.1", mapped,
                          component_roles=tuple(roles))
