"""Mapped acyl pseudohalide construction under P-65.5.2."""

from dataclasses import dataclass
from collections import Counter
from itertools import permutations

from .molecule_iteration import iter_atoms
from .component_mapping import MappedComponent
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class _PseudoGroup:
    kind: str
    root: int
    atoms: frozenset[int]


_SINGLE = 1.0
_DOUBLE = 2.0
_TRIPLE = 3.0
_SENIORITY_ORDER = {name: position for position, name in enumerate((
    "azide", "cyanide", "isocyanide", "isocyanate", "isothiocyanate",
    "isoselenocyanate", "isotellurocyanate",
))}
_CLASS_WORDS = (
    "azide", "cyanide", "isocyanide", "isocyanate", "isothiocyanate",
    "isoselenocyanate", "isotellurocyanate",
)
_CITATION_ORDER = {name: position for position, name in enumerate(sorted(_CLASS_WORDS))}
_MULTIPLIERS = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa",
                7: "hepta", 8: "octa", 9: "nona", 10: "deca"}


def _bond_order(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond.GetBondTypeAsDouble() if bond is not None else 0.0


def _neighbors(mol, idx):
    return tuple(atom.GetIdx() for atom in mol.GetAtomWithIdx(idx).GetNeighbors())


def _exact_neighbors(mol, idx, expected):
    return set(_neighbors(mol, idx)) == set(expected)


def _plain_atom(mol, idx, atomic_num, charge=0):
    atom = mol.GetAtomWithIdx(idx)
    return (atom.GetAtomicNum() == atomic_num and atom.GetFormalCharge() == charge
            and atom.GetNumRadicalElectrons() == 0 and not atom.GetIsotope()
            and int(atom.GetChiralTag()) == 0)


def _motif_atom(mol, idx, atomic_num, charges):
    atom = mol.GetAtomWithIdx(idx)
    return (atom.GetAtomicNum() == atomic_num and atom.GetFormalCharge() in charges
            and atom.GetNumRadicalElectrons() == 0 and not atom.GetIsotope()
            and int(atom.GetChiralTag()) == 0)


def _validate_cyanide(mol, center, root):
    if not _plain_atom(mol, root, 6) or _bond_order(mol, center, root) != _SINGLE:
        return None
    ns = _neighbors(mol, root)
    terminal = [i for i in ns if i != center]
    if len(terminal) != 1 or not _exact_neighbors(mol, root, (center, terminal[0])):
        return None
    nitrogen = terminal[0]
    if (not _plain_atom(mol, nitrogen, 7) or _neighbors(mol, nitrogen) != (root,)
            or _bond_order(mol, root, nitrogen) != _TRIPLE):
        return None
    if mol.GetAtomWithIdx(root).GetTotalNumHs(includeNeighbors=True):
        return None
    return _PseudoGroup("cyanide", root, frozenset({root, nitrogen}))


def _validate_isocyanide(mol, center, root):
    # The accepted neutral R-CO-NC valence form is R-CO-[N+]#[C-].
    if not _plain_atom(mol, root, 7, 1) or _bond_order(mol, center, root) != _SINGLE:
        return None
    ns = [i for i in _neighbors(mol, root) if i != center]
    if len(ns) != 1 or not _exact_neighbors(mol, root, (center, ns[0])):
        return None
    carbon = ns[0]
    if (not _plain_atom(mol, carbon, 6, -1) or _neighbors(mol, carbon) != (root,)
            or _bond_order(mol, root, carbon) != _TRIPLE):
        return None
    return _PseudoGroup("isocyanide", root, frozenset({root, carbon}))


def _validate_azide(mol, center, root):
    if not _motif_atom(mol, root, 7, {0, -1}) or _bond_order(mol, center, root) != _SINGLE:
        return None
    first = [i for i in _neighbors(mol, root) if i != center]
    if len(first) != 1:
        return None
    middle = first[0]
    if not _plain_atom(mol, middle, 7, 1):
        return None
    if not _exact_neighbors(mol, root, (center, middle)):
        return None
    terminal = [i for i in _neighbors(mol, middle) if i != root]
    if len(terminal) != 1:
        return None
    last = terminal[0]
    if not _exact_neighbors(mol, middle, (root, last)) or not _motif_atom(mol, last, 7, {0, -1}):
        return None
    # Two common neutral resonance forms are accepted. Internal charge must
    # cancel exactly; no charge is tolerated outside this three-nitrogen unit.
    first_order = _bond_order(mol, root, middle)
    last_order = _bond_order(mol, middle, last)
    charges = tuple(mol.GetAtomWithIdx(i).GetFormalCharge() for i in (root, middle, last))
    valid = ((first_order == _DOUBLE and last_order == _DOUBLE and charges == (0, 1, -1))
             or (first_order == _SINGLE and last_order == _TRIPLE and charges == (-1, 1, 0)))
    if not valid or sum(charges) != 0:
        return None
    if any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)
           for i in (root, middle, last)):
        return None
    return _PseudoGroup("azide", root, frozenset({root, middle, last}))


def _validate_isocyanate(mol, center, root):
    if not _plain_atom(mol, root, 7) or _bond_order(mol, center, root) != _SINGLE:
        return None
    roots = [i for i in _neighbors(mol, root) if i != center]
    if len(roots) != 1:
        return None
    carbon = roots[0]
    if not _plain_atom(mol, carbon, 6) or not _exact_neighbors(mol, root, (center, carbon)):
        return None
    if _bond_order(mol, root, carbon) != _DOUBLE:
        return None
    terminals = _neighbors(mol, carbon)
    if len(terminals) != 2 or root not in terminals:
        return None
    terminal = next(i for i in terminals if i != root)
    terminal_num = mol.GetAtomWithIdx(terminal).GetAtomicNum()
    kind = {8: "isocyanate", 16: "isothiocyanate", 34: "isoselenocyanate",
            52: "isotellurocyanate"}.get(terminal_num)
    if (kind is None or not _plain_atom(mol, terminal, terminal_num)
            or _neighbors(mol, terminal) != (carbon,)
            or _bond_order(mol, carbon, terminal) != _DOUBLE):
        return None
    if any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) for i in (root, carbon, terminal)):
        return None
    return _PseudoGroup(kind, root, frozenset({root, carbon, terminal}))


def _pseudohalide_at(mol, center, root):
    validators = (_validate_cyanide, _validate_isocyanide,
                  _validate_azide, _validate_isocyanate)
    matches = [match for validator in validators if (match := validator(mol, center, root))]
    return matches[0] if len(matches) == 1 else None


def _carbonyl_candidates(mol):
    candidates = []
    for atom in iter_atoms(mol):
        center = atom.GetIdx()
        if atom.GetAtomicNum() != 6:
            continue
        carbonyl_oxygens = [i for i in _neighbors(mol, center)
                            if mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                            and _bond_order(mol, center, i) == _DOUBLE]
        if len(carbonyl_oxygens) != 1:
            continue
        for root in _neighbors(mol, center):
            if root == carbonyl_oxygens[0] or _bond_order(mol, center, root) != _SINGLE:
                continue
            pseudo = _pseudohalide_at(mol, center, root)
            if pseudo:
                candidates.append((center, carbonyl_oxygens[0], pseudo))
    return candidates


def _principal_survives_after_pseudo_removal(mol, Chem, center, pseudo_atoms):
    """Expose all other groups after replacing this pseudo unit by H.

    The probe is used only to decide whether another recognized principal
    class competes. Its naming output, atom order, and stereo are never used.
    """
    rw = Chem.RWMol(mol)
    for idx in sorted(pseudo_atoms, reverse=True):
        rw.RemoveAtom(idx)
    remapped_center = center - sum(idx < center for idx in pseudo_atoms)
    hydrogen = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(remapped_center, hydrogen, Chem.BondType.SINGLE)
    probe = rw.GetMol()
    try:
        Chem.SanitizeMol(probe)
    except (ValueError, RuntimeError):
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(probe)
    if not inventory.supported or not inventory.groups:
        return False
    # The H cap turns this acyl site into an aldehyde only for recognition.
    # Its temporary class 15 cannot set the original pseudohalide's priority
    # (class 10). Amides/nitriles/aldehydes may be subordinate; another acid,
    # ester, or class-10 site needs a different complete construction.
    return not any(group.center != remapped_center and group.seniority[0] <= 10
                   for group in inventory.groups)


def acyl_pseudohalide_decision(mol, Chem):
    """Return a mapped functional-class decision for acyl pseudohalides.

    P-65.5.2 support is limited to fully mapped acid derivatives. Exact
    pseudohalide motifs are recognized from atom identity, bond order and
    charge pattern; names are produced from typed group roles.
    """
    if mol.GetNumAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups():
        return None
    if (any(a.GetNumRadicalElectrons() for a in iter_atoms(mol))
            or any(int(a.GetChiralTag()) and a.GetAtomicNum() in {7, 8, 16, 34, 52}
                   for a in iter_atoms(mol))):
        return None
    if any(a.GetIsotope() and a.GetAtomicNum() != 6 for a in iter_atoms(mol)):
        return None

    candidates = _carbonyl_candidates(mol)
    if len(candidates) > 1:
        return _multisite_decision(mol, Chem, candidates)
    if len(candidates) != 1:
        return None
    center, carbonyl_oxygen, pseudo = candidates[0]
    heavy_atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    pseudo_atoms = pseudo.atoms
    # Pseudohalide class atoms have no established isotope descriptor locants.
    # Carbon-13 annotations on the mapped acyl component remain supported.
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in pseudo_atoms):
        return None
    if any(a.GetFormalCharge() for a in iter_atoms(mol) if a.GetIdx() not in pseudo_atoms):
        return None
    pseudo_charge = sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in pseudo_atoms)
    if pseudo_charge != 0 or any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                                  for i in pseudo_atoms):
        return None
    # Isotope labels on the carbonyl oxygen are not covered by P-82.6.4 for
    # pseudohalide endings, even though the shared acid component can render
    # that role for other derivative classes.
    if mol.GetAtomWithIdx(carbonyl_oxygen).GetIsotope():
        return None

    # Parent groups with higher P-41 seniority keep the principal route. The
    # shared acid mapper also rejects a second suffix-acid site after capping.
    if not _principal_survives_after_pseudo_removal(mol, Chem, center, pseudo_atoms):
        return None
    from .acid_derivative_methods import _acyl, _acid_component
    part = _acid_component(mol, Chem, center, pseudo.root)
    if part is None or part.atoms & pseudo_atoms:
        return None
    acyl = _acyl(part.name)
    if not acyl:
        return None
    acid_atoms = frozenset(i for i in part.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    if acid_atoms & pseudo_atoms or acid_atoms | pseudo_atoms != heavy_atoms:
        return None
    name = {"azide": "azide", "cyanide": "cyanide", "isocyanide": "isocyanide",
            "isocyanate": "isocyanate", "isothiocyanate": "isothiocyanate",
            "isoselenocyanate": "isoselenocyanate", "isotellurocyanate": "isotellurocyanate"}[pseudo.kind]
    return MethodDecision(acyl + " " + name, "functional-class",
                          frozenset({center}), heavy_atoms,
                          "P-65.5.2.1/P-65.1.7.1/P-65.1.7.2/P-65.1.7.4",
                          part.atom_locants,
                          component_roles=(("acyl", acid_atoms, part.atom_locants),
                                           ("pseudohalide:" + pseudo.kind, pseudo_atoms, ())))


def _multisite_decision(mol, Chem, candidates):
    """Mapped all-site construction for the P-65.5.2.1 multisite rule."""
    if len(candidates) < 2:
        return None
    if any(a.IsInRing() and int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    centers = [center for center, _, _ in candidates]
    pseudos = [pseudo for _, _, pseudo in candidates]
    if len(set(centers)) != len(centers):
        return None  # Carbonic-acid di-derivatives use P-65.5.3.
    pseudo_atoms = set()
    for pseudo in pseudos:
        if pseudo_atoms & set(pseudo.atoms):
            return None
        pseudo_atoms.update(pseudo.atoms)
    pseudo_atoms = frozenset(pseudo_atoms)
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if not pseudo_atoms <= heavy or any(
        mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and mol.GetAtomWithIdx(i).GetIsotope()
        for i in heavy - pseudo_atoms
    ):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in pseudo_atoms):
        return None
    if any(a.GetFormalCharge() for a in iter_atoms(mol) if a.GetIdx() not in pseudo_atoms):
        return None
    if any(sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in pseudo.atoms) != 0
           for pseudo in pseudos):
        return None
    if any(mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
           or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in pseudo_atoms):
        return None
    if not _class_word_order_is_source_supported(pseudos):
        return None
    if not _unlike_assignments_are_graph_invariant(mol, Chem, candidates):
        return None
    if not _principal_survives_after_all_pseudo_removal(mol, Chem, candidates,
                                                        pseudo_atoms):
        return None

    # Remove each complete validated pseudohalide and cap its acyl carbonyl
    # with a temporary OH. The shared parent selector sees all acid sites
    # together, while original indices remain available for role translation.
    retained = [i for i in range(mol.GetNumAtoms()) if i not in pseudo_atoms]
    old_to_new = {old: new for new, old in enumerate(retained)}
    rw = Chem.RWMol(mol)
    for idx in sorted(pseudo_atoms, reverse=True):
        rw.RemoveAtom(idx)
    cap_atoms = set()
    for center in centers:
        if center not in old_to_new:
            return None
        cap = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(old_to_new[center], cap, Chem.BondType.SINGLE)
        cap_atoms.add(cap)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    parent_graph = Chem.Mol(capped)
    for atom in parent_graph.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in parent_graph.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistry(parent_graph, cleanIt=True, force=True)
    from .cross_route_selection import cross_route_selection
    selection = cross_route_selection(parent_graph, Chem)
    if selection is None or selection.selected.principal_kind != "acid":
        return None
    construction = selection.selected.construction
    if construction is None or not construction.parent.endswith(" acid"):
        return None

    new_to_old = {new: old for old, new in old_to_new.items()}
    cap_atoms = frozenset(cap_atoms)
    mapping = {new_to_old[idx]: loc for idx, loc in construction.atom_locants
               if idx in new_to_old}
    owned = {new_to_old[idx] for idx in construction.atoms if idx in new_to_old}
    embedded = {new_to_old[idx] for idx in construction.embedded_atoms
                if idx in new_to_old}
    # Ring carboxyl suffix centers may be exposed only on ParentProposal.
    for idx, loc in selection.selected.atom_locants:
        if idx in new_to_old:
            original = new_to_old[idx]
            if original not in embedded:
                mapping.setdefault(original, loc)
                owned.add(original)
    mapped = MappedComponent(
        construction.parent,
        tuple(mapping.items()), frozenset(owned), construction.prefixes,
        frozenset(embedded), construction.hydro_prefix,
        construction.systematic_parent, construction.nitrogen_site_locants,
        construction.isotope_unlocanted_elements, construction.prefix_mode,
        frozenset(i for i in construction.local_isotope_atoms if i in owned),
        construction.omit_methyl_prefix_locant,
    )
    if mapped.atoms & mapped.embedded_atoms or not mapped.atoms <= set(mapping):
        return None
    from .component_mapping import decorate_component, has_annotations
    if has_annotations(mol, mapped.embedded_atoms):
        return None
    if set(mapped.atoms) | set(mapped.embedded_atoms) != set(heavy - pseudo_atoms):
        return None
    acid_name = decorate_component(mol, Chem, mapped)
    if not acid_name:
        return None
    from .acid_derivative_methods import _acyl
    acyl_name = _pseudohalide_acyl_name(acid_name, _acyl)
    if not acyl_name:
        return None
    class_terms = _pseudohalide_class_terms([pseudo.kind for pseudo in pseudos])
    if not class_terms:
        return None
    roles = [("acyl", frozenset(heavy - pseudo_atoms), tuple(mapping.items()))]
    for kind in sorted({_kind for _kind in (p.kind for p in pseudos)},
                       key=lambda item: _CITATION_ORDER[item]):
        atoms = frozenset().union(*(p.atoms for p in pseudos if p.kind == kind))
        roles.append(("pseudohalide:" + kind, atoms, ()))
    return MethodDecision(
        acyl_name + " " + class_terms, "functional-class",
        frozenset(centers), heavy,
        "P-65.5.2.1/P-65.1.7.1/P-65.1.7.2/P-65.1.7.4",
        tuple(mapping.items()), component_roles=tuple(roles),
    )


def _principal_survives_after_all_pseudo_removal(mol, Chem, candidates, pseudo_atoms):
    rw = Chem.RWMol(mol)
    for idx in sorted(pseudo_atoms, reverse=True):
        rw.RemoveAtom(idx)
    remapped = {center: center - sum(idx < center for idx in pseudo_atoms)
                for center, _, _ in candidates}
    for center in remapped.values():
        hydrogen = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(center, hydrogen, Chem.BondType.SINGLE)
    probe = rw.GetMol()
    try:
        Chem.SanitizeMol(probe)
    except (ValueError, RuntimeError):
        return False
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(probe)
    if not inventory.supported or not inventory.groups:
        return False
    synthetic_centers = set(remapped.values())
    return not any(group.center not in synthetic_centers and group.seniority[0] <= 10
                   for group in inventory.groups)


def _pseudohalide_class_terms(kinds):
    counts = Counter(kinds)
    terms = []
    for kind in sorted(counts, key=lambda item: _CITATION_ORDER[item]):
        count = counts[kind]
        multiplier = _MULTIPLIERS.get(count, "")
        if count > 1 and not multiplier:
            return ""
        terms.append(multiplier + kind)
    return " ".join(terms)


def _class_word_order_is_source_supported(pseudos):
    """Do not infer citation order from the P-65.5.2.1 seniority table.

    P-15.2.1.2 gives alphanumerical order for different functional-class
    groups. P-65.5.2.1 gives a separate seniority order, without stating that
    it is citation order. Admit a mixture only where both criteria coincide;
    equal classes and the source-spelled NC/NCS pair are directly supported.
    """
    kinds = {pseudo.kind for pseudo in pseudos}
    if len(kinds) <= 1:
        return True
    if kinds == {"isocyanide", "isothiocyanate"}:
        return True  # exact unlike sequence shown in the P-65.5.2.1 example
    alpha = sorted(kinds, key=lambda item: _CITATION_ORDER[item])
    seniority = sorted(kinds, key=lambda item: _SENIORITY_ORDER[item])
    return alpha == seniority


def _unlike_assignments_are_graph_invariant(mol, Chem, candidates):
    """Require every unlike-class placement to be the same mapped structure.

    P-14.3.4.4 does not permit omitting locants where interchange of unlike
    groups can make an isomer. We test the capped acid skeleton's actual
    stereochemical/isotopic automorphisms; no locant syntax is inferred here.
    The number of distinct assignments is bounded to keep the proof finite.
    """
    kinds = tuple(pseudo.kind for _, _, pseudo in candidates)
    if len(set(kinds)) <= 1:
        return True
    if len(kinds) > 6:
        return False
    roots = [pseudo.root for _, _, pseudo in candidates]
    centers = [center for center, _, _ in candidates]
    root_to_remove = {idx for _, _, pseudo in candidates
                      for idx in pseudo.atoms if idx != pseudo.root}
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    for idx in sorted(root_to_remove, reverse=True):
        rw.RemoveAtom(idx)
    old_to_new = {atom.GetAtomMapNum() - 1: atom.GetIdx() for atom in rw.GetAtoms()}
    for root in roots:
        atom = rw.GetAtomWithIdx(old_to_new[root])
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNoImplicit(True)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    core = rw.GetMol()
    try:
        Chem.SanitizeMol(core)
    except (ValueError, RuntimeError):
        return False
    # Keep the core's original tetrahedral parity while ignoring CIP labels,
    # which can change if the hypothetical class placement changes.
    matches = core.GetSubstructMatches(core, uniquify=False, useChirality=True,
                                       maxMatches=10000)
    center_indices = tuple(old_to_new[idx] for idx in centers)
    center_position = {idx: pos for pos, idx in enumerate(center_indices)}
    automorphisms = []
    for match in matches:
        mapped = []
        for idx in center_indices:
            target = match[idx]
            if target not in center_position:
                break
            mapped.append(center_position[target])
        if len(mapped) == len(center_indices):
            automorphisms.append(tuple(mapped))
    if not automorphisms:
        return False
    assignments = set(permutations(kinds))
    return all(any(all(kinds[source] == assignment[target]
                       for source, target in enumerate(mapping))
                   for mapping in automorphisms)
               for assignment in assignments)


def _pseudohalide_acyl_name(acid_name, acyl_builder):
    if acid_name.endswith("oxalic acid"):
        return acid_name[:-len("oxalic acid")] + "oxalyl"
    if acid_name.endswith("ethanedioic acid"):
        return acid_name[:-len("ethanedioic acid")] + "oxalyl"
    return acyl_builder(acid_name)
