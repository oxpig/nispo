"""Mapped substitutive names for acid salts of polybasic organic acids.

P-65.6.2.3.1 prefers the acid-as-prefix construction for organic acid salts.
This producer builds only a bounded, fully mapped component; ionic assembly
and cation stoichiometry remain the caller's responsibility.
"""

from dataclasses import dataclass, replace
from collections import Counter
import re

from .component_mapping import MappedComponent, decorate_component, has_annotations
from .extended_inventory import selection_inventory


@dataclass(frozen=True, slots=True)
class AcidSaltComponent:
    """A named organic anion with original graph ownership and charge."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    substituted: bool = False


def acid_salt_component(mol, Chem, *, include_fully_deprotonated=True):
    """Return a mapped polycarboxylate component name, or ``None``.

    Partial salts use the P-65.6.2.3.1 substitutive method: maximize anionic
    suffix groups on the shared parent, cite residual acids as ``carboxy``
    prefixes, and any unselected anions as ``carboxylato`` prefixes.
    Fully deprotonated components
    are admitted only through the existing complete mapped parent renderer.
    """
    if mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups():
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    if any(a.GetIsotope() and a.GetAtomicNum() != 6 for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None

    inventory = selection_inventory(mol)
    sites = tuple(g for g in inventory.groups if g.kind in {"acid", "carboxylate"})
    if len(sites) < 2 or len(sites) > 6 or len(sites) != len(inventory.groups):
        return None
    anions = tuple(g for g in sites if g.kind == "carboxylate")
    acids = tuple(g for g in sites if g.kind == "acid")
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if charge != -len(anions):
        return None
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    if charged != {g.ligand for g in anions} or any(
        mol.GetAtomWithIdx(i).GetFormalCharge() != -1 for i in charged
    ):
        return None
    if any(g.owner < 0 or mol.GetAtomWithIdx(g.owner).GetAtomicNum() != 6 for g in sites):
        return None

    rings = mol.GetRingInfo().AtomRings()
    if len(rings) > 1 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                             for ring in rings for i in ring):
        return None

    if not acids:
        if not include_fully_deprotonated:
            return None
        from .complete_functional_parent import functional_parent_candidates
        candidates = functional_parent_candidates(
            mol, Chem, include_single=True, principal=anions,
            mapped_ring_prefixes=True,
        )
        for candidate in candidates:
            construction = candidate.construction
            if construction is None:
                continue
            if has_annotations(mol, construction.embedded_atoms):
                continue
            mapped = tuple((i, loc) for i, loc in construction.atom_locants)
            if not _covers_component(mol, Chem, construction.atoms,
                                     construction.embedded_atoms):
                continue
            return AcidSaltComponent(
                candidate.name, frozenset(a.GetIdx() for a in mol.GetAtoms()
                                           if a.GetAtomicNum() > 1), mapped, charge,
                "P-65.6.2.1/P-77.3.1",
                (("route", "complete-functional"),
                 ("anion_sites", tuple((g.center, g.owner) for g in anions)),
                 ("parent", candidate.parent_atoms)),
                substituted=bool(construction.prefixes),
            )
        return None

    return _partial_candidate(mol, Chem, sites, anions, acids, charge)


def _partial_candidate(mol, Chem, sites, anions, acids, charge):
    # Keep every anionic site on one shared parent graph so P-44.1.1 can
    # maximize principal-group suffix coverage. Free acids are separate
    # substituent units under P-65.6.2.3.1 and are restored as mapped prefixes.
    remove = {i for group in acids for i in group.atoms}
    # No explicit H atoms are expected in this bounded valence representation.
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in range(mol.GetNumAtoms())):
        return None
    retained = [i for i in range(mol.GetNumAtoms()) if i not in remove]
    old_to_new = {old: new for new, old in enumerate(retained)}
    rw = Chem.RWMol(mol)
    caps = Counter(group.owner for group in acids)
    for owner, count in caps.items():
        if (owner in remove or any(
                mol.GetBondBetweenAtoms(owner, group.center).GetBondTypeAsDouble() != 1
                for group in acids if group.owner == owner)):
            return None
        atom = rw.GetAtomWithIdx(owner)
        # Bracketed/isotopic carbon has explicit H counts. Removing an acid
        # branch otherwise leaves a radical in the temporary parent graph.
        if atom.GetNoImplicit() or atom.GetNumExplicitHs():
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + count)
    for i in sorted(remove, reverse=True):
        rw.RemoveAtom(i)
    for target in anions:
        if target.ligand not in old_to_new:
            return None
        hydroxyl = rw.GetAtomWithIdx(old_to_new[target.ligand])
        hydroxyl.SetFormalCharge(0)
        hydroxyl.SetNoImplicit(False)
        hydroxyl.SetNumExplicitHs(0)
    normalized = rw.GetMol()
    # This capped graph supplies only the constitutional parent. The final
    # mapped construction and numbering use the original isotope/CIP graph.
    Chem.RemoveStereochemistry(normalized)
    try:
        Chem.SanitizeMol(normalized)
    except (ValueError, RuntimeError):
        return None

    from .cross_route_selection import cross_route_selection
    from .acid_derivative_methods import _capped_acid_selection
    selection = _capped_acid_selection(
        Chem, normalized.ToBinary(Chem.PropertyPickleOptions.AllProps),
        cross_route_selection)
    if selection is None or selection.selected.principal_kind != "acid":
        return None
    selected_construction = selection.selected.construction
    if selected_construction is None:
        return None
    new_to_old = {new: old for old, new in old_to_new.items()}
    suffix_centers = frozenset(new_to_old[i] for i in selection.selected.principal_centers
                               if i in new_to_old)
    translated = _translate_construction(selected_construction, new_to_old)
    if translated is None:
        return None
    # Ring carboxyl suffix centers are kept outside the ring skeleton by the
    # shared renderer. ParentProposal carries their mapped C-locants; bring
    # those original atoms into the typed component ownership explicitly.
    proposal_mapping = {new_to_old[idx]: loc
                        for idx, loc in selection.selected.atom_locants
                        if idx in new_to_old}
    missing_centers = set(suffix_centers) - set(dict(translated.atom_locants))
    if not missing_centers <= proposal_mapping.keys():
        return None
    translated = replace(
        translated,
        atom_locants=translated.atom_locants + tuple(
            (idx, proposal_mapping[idx]) for idx in sorted(missing_centers)
        ),
        atoms=translated.atoms | missing_centers,
        embedded_atoms=translated.embedded_atoms - missing_centers,
    )
    constructions = _parent_constructions(mol, translated)
    candidates = []
    for construction in constructions:
        parent = _anion_parent(construction.parent)
        if not parent:
            continue
        candidates.append(_render_partial_construction(
            mol, Chem, sites, anions, acids, charge, selection, suffix_centers,
            construction, parent,
        ))
    candidates = [candidate for candidate in candidates if candidate is not None]
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def _render_partial_construction(mol, Chem, sites, anions, acids, charge, selection,
                                 suffix_centers, construction, parent):
    mapping = dict(construction.atom_locants)
    owned = set(construction.atoms)
    embedded = set(construction.embedded_atoms)
    prefixes = list(construction.prefixes)
    site_proof = []
    for group in sites:
        locant = mapping.get(group.owner)
        selected_suffix = (group in anions and group.center in suffix_centers
                           and group.center in mapping)
        if not selected_suffix:
            if not locant:
                branch = _acid_site_branch(mol, Chem, group, mapping)
                if branch is None:
                    return None
                anchor_locant, text, branch_atoms, previous_text = branch
                matches = [i for i, row in enumerate(prefixes)
                           if row == (anchor_locant, previous_text)]
                if len(matches) != 1:
                    return None
                prefixes[matches[0]] = (anchor_locant, text)
                embedded.update(branch_atoms)
                site_proof.append((group.center, group.owner, group.kind,
                                   anchor_locant, "branch"))
                continue
            prefix = "carboxy" if group.kind == "acid" else "carboxylato"
            prefixes.append((locant, prefix))
            embedded.update(group.atoms)
        site_proof.append((group.center, group.owner, group.kind, locant))

    suffix_centers = set(suffix_centers)
    if not suffix_centers or not suffix_centers <= set(mapping):
        return None
    if owned & embedded or not owned <= mapping.keys():
        return None
    # Prefix branches are named textually; until their local isotope/stereo
    # construction is available, decline annotations that would be lost there.
    if has_annotations(mol, embedded):
        return None
    if not _covers_component(mol, Chem, owned, embedded):
        return None
    from .retained_acid_rules import retained_acid_locant_evidence

    retained_evidence = retained_acid_locant_evidence(
        mol, Chem, parent_class="acetic acid" if parent == "acetate" else "",
        principal_kind="carboxylate", parent_atoms=frozenset(owned),
        atom_locants=tuple(mapping.items()), prefixes=tuple(prefixes),
        isotope_forces_locants=any(atom.GetIsotope() for atom in mol.GetAtoms()),
    )
    translated = MappedComponent(
        parent, tuple(mapping.items()), frozenset(owned), tuple(prefixes),
        frozenset(embedded), construction.hydro_prefix,
        construction.systematic_parent, construction.nitrogen_site_locants,
        construction.isotope_unlocanted_elements, construction.prefix_mode,
        frozenset(i for i in construction.local_isotope_atoms if i in owned),
        construction.omit_methyl_prefix_locant,
        retained_acid_locant_evidence=retained_evidence,
    )
    name = decorate_component(mol, Chem, translated)
    if not name or len(name) > 260:
        return None
    from .parent_annotations import isotope_seniority, stereo_numbering
    isotope = isotope_seniority(mol, mapping)
    stereo = stereo_numbering(mol, mapping)
    if isotope is None or stereo is None:
        return None
    suffix_locants = tuple(sorted(
        int(mapping[g.center]) if mapping[g.center].isdigit()
        else int(mapping[g.owner])
        for g in anions if g.center in suffix_centers
        and (mapping[g.center].isdigit() or mapping[g.owner].isdigit())
    ))
    prefix_locants = tuple(sorted(int(loc) for loc, _ in prefixes if loc.isdigit()))
    key = (suffix_locants, prefix_locants, isotope, stereo)
    proof = (("route", selection.selected.route),
             ("suffix_centers", tuple(sorted(suffix_centers))),
             ("sites", tuple(site_proof)),
             ("mapped_parent", tuple(sorted(mapping.items()))),
             ("retained_locant_omission", retained_evidence is not None),
             ("countercharge", charge))
    return key, AcidSaltComponent(
        name, frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
        tuple(mapping.items()), charge,
        "P-65.6.2.3.1/P-77.3.2", proof,
        substituted=bool(prefixes),
    )


def _translate_construction(construction, new_to_old):
    if any(i not in new_to_old for i in construction.atoms | construction.embedded_atoms):
        return None
    return replace(
        construction,
        atom_locants=tuple((new_to_old[i], loc) for i, loc in construction.atom_locants
                           if i in new_to_old),
        atoms=frozenset(new_to_old[i] for i in construction.atoms),
        embedded_atoms=frozenset(new_to_old[i] for i in construction.embedded_atoms),
        local_isotope_atoms=frozenset(new_to_old[i] for i in construction.local_isotope_atoms
                                      if i in new_to_old),
        # Evidence on the capped graph has different atom IDs and prefix
        # roles. Rebuild it after restoring acid sites on the original graph.
        retained_acid_locant_evidence=None,
    )


def _parent_constructions(mol, construction):
    """Enumerate mapped chain/ring directions until omitted acid prefixes land."""
    mapping = dict(construction.atom_locants)
    skeleton = {idx for idx, loc in mapping.items()
                if loc.isdigit() and mol.GetAtomWithIdx(idx).GetAtomicNum() == 6}
    if len(skeleton) < 2:
        return (construction,)
    adjacency = {idx: set() for idx in skeleton}
    for idx in skeleton:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = atom.GetIdx()
            if other in skeleton:
                adjacency[idx].add(other)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return (construction,)
    ring = all(mol.GetAtomWithIdx(i).IsInRing() for i in skeleton)
    if ring:
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return (construction,)
        paths = []
        start = min(skeleton)
        for neighbor in sorted(adjacency[start]):
            path = [start, neighbor]
            while len(path) < len(skeleton):
                following = adjacency[path[-1]] - {path[-2]}
                nxt = next(iter(following - {start}), None)
                if nxt is None:
                    break
                path.append(nxt)
            if len(path) == len(skeleton):
                paths.append(tuple(path))
        variants = []
        for path in paths:
            variants.extend(tuple(path[i:] + path[:i]) for i in range(len(path)))
    else:
        endpoints = [idx for idx in skeleton if len(adjacency[idx]) <= 1]
        if len(endpoints) != 2 or any(len(adjacency[i]) == 0 for i in skeleton):
            return (construction,)
        path = [min(endpoints)]
        while len(path) < len(skeleton):
            options = adjacency[path[-1]] - set(path)
            if len(options) != 1:
                return (construction,)
            path.append(next(iter(options)))
        variants = [tuple(path), tuple(reversed(path))]

    old_numeric = {loc: idx for idx, loc in mapping.items() if idx in skeleton}
    options = []
    for path in variants:
        new_mapping = dict(mapping)
        positions = {idx: str(pos) for pos, idx in enumerate(path, 1)}
        for idx in skeleton:
            new_mapping[idx] = positions[idx]
        for idx, loc in mapping.items():
            if idx in skeleton:
                continue
            match = re.fullmatch(r"(.*?)(\d+)(.*)", loc)
            if match and match[2] in old_numeric:
                anchor = old_numeric[match[2]]
                new_mapping[idx] = match[1] + positions[anchor] + match[3]
        old_to_new_loc = {old: new_mapping[idx] for old, idx in old_numeric.items()}
        prefixes = []
        for locant, text in construction.prefixes:
            if locant not in old_to_new_loc:
                return (construction,)
            prefixes.append((old_to_new_loc[locant], text))
        options.append(replace(
            construction,
            atom_locants=tuple(new_mapping.items()),
            prefixes=tuple(prefixes),
        ))
    return tuple(options) or (construction,)


def _acid_site_branch(mol, Chem, group, mapping):
    """Restore an acid site whose owner was a branch in the anionic parent."""
    from .component_mapping import mapped_branch

    owner = group.owner
    anchors = [a.GetIdx() for a in mol.GetAtomWithIdx(owner).GetNeighbors()
               if a.GetIdx() in mapping and a.GetIdx() not in group.atoms]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    text, atoms = mapped_branch(mol, Chem, owner, anchor)
    if not text or not set(group.atoms) <= atoms:
        return None
    retained = [i for i in range(mol.GetNumAtoms()) if i not in group.atoms]
    old_to_new = {old: new for new, old in enumerate(retained)}
    if owner not in old_to_new or anchor not in old_to_new:
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted(group.atoms, reverse=True):
        rw.RemoveAtom(idx)
    try:
        capped = rw.GetMol()
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    # Removing the acid unit recovers the exact prefix token used by the
    # shared parent proposal. Replace that token with the complete mapped
    # carboxy/carboxylato branch, retaining its parent attachment locant.
    capped_text, _ = mapped_branch(capped, Chem, old_to_new[owner], old_to_new[anchor])
    if not capped_text:
        return None
    anchor_locant = mapping.get(anchor)
    if anchor_locant is None:
        return None
    return anchor_locant, text, atoms, capped_text


def _anion_parent(parent):
    """Change one typed organic acid ending to its corresponding anion term."""
    for acid, anion in (("benzoic acid", "benzoate"),
                        ("acetic acid", "acetate"),
                        ("formic acid", "formate")):
        if parent.endswith(acid):
            return parent[:-len(acid)] + anion
    if parent.endswith("carboxylic acid"):
        return parent[:-len("carboxylic acid")] + "carboxylate"
    if parent.endswith("ic acid"):
        return parent[:-len("ic acid")] + "ate"
    return ""


def _covers_component(mol, Chem, atoms, embedded):
    owned = set(atoms) | set(embedded)
    expected = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != expected or set(atoms) & set(embedded):
        return False
    from .stereochemistry import _indexed_bond_stereochemistry
    if any({a, b} & set(embedded) for a, b, _ in _indexed_bond_stereochemistry(mol, Chem)):
        return False
    return True
