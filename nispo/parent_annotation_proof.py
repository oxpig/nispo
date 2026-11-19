"""Immutable mapped proof for the bounded annotated monocycle frontier.

The proof keeps graph/stereogenic-feature census separate from descriptor
serialization. Unknown configuration is admitted only when the feature is
proved to have the ordinary capital R/S class; isotope-created units and
unclassified features remain barriers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product

from .molecule_cache import exact_graph_computation


@dataclass(frozen=True, slots=True)
class _SourceInventoryEvidence:
    """Fresh original-graph identities, independent of parent numbering."""

    supported: bool
    group_ids: frozenset = frozenset()
    source_status: str | None = None
    source_ids: frozenset = frozenset()
    principal_ids: frozenset = frozenset()


class ParentAnnotationStatus(str, Enum):
    PROVED = "proved"
    OUT_OF_SCOPE = "out_of_scope"
    BLOCKED = "blocked"


class PotentialStereoTriggerStatus(str, Enum):
    NO_TRIGGER = "no_trigger"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"
    OUTSIDE_PROFILE = "outside_profile"


@dataclass(frozen=True, slots=True)
class PotentialStereoTrigger:
    status: PotentialStereoTriggerStatus
    ring_atoms: frozenset[int] = frozenset()
    feature_census: tuple = ()
    triggering_atoms: tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ParentStereoFeature:
    atom_id: int
    feature_type: str
    state: str
    descriptor_class: str
    descriptor: str = ""
    locant: str = ""


@dataclass(frozen=True, slots=True)
class ParentAnnotationProof:
    status: ParentAnnotationStatus
    original_graph_signature: tuple = ()
    principal_identity: tuple = ()
    principal_class: str = ""
    parent_atoms: frozenset[int] = frozenset()
    parent_locants: tuple[tuple[int, str], ...] = ()
    suffix_core_ownership: tuple = ()
    isotope_ownership: tuple[tuple[int, int, int, str], ...] = ()
    original_feature_census: tuple = ()
    isotope_cleared_feature_census: tuple = ()
    stereo_features: tuple[ParentStereoFeature, ...] = ()
    source_rule_order: tuple[str, ...] = ()
    reason: str = ""
    n_branch_ownership: tuple = ()

    @property
    def unknown_stereo_locants(self):
        return tuple(sorted(
            (feature.locant for feature in self.stereo_features
             if feature.state == "unknown_ordinary"),
            key=_locant_sort_key,
        ))


def implicit_unknown_monocycle_trigger(mol, Chem, *, principal_groups,
                                       principal_class):
    """Census implicit unknown units for the bounded suffix-ring profile.

    This is an admission trigger only. It never assigns a configuration or
    supplies parent locants; the full mapped proof remains responsible for
    those decisions. An empty census is distinct from a census failure or an
    unsupported feature in an otherwise qualified source ring.
    """
    groups = tuple(principal_groups)
    if (principal_class not in {"acid", "amide", "nitrile"} or not groups
            or any(group.kind != principal_class for group in groups)):
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.OUTSIDE_PROFILE,
            reason="Principal class is outside the source-qualified ring trigger.",
        )
    rings = tuple(tuple(int(i) for i in ring) for ring in mol.GetRingInfo().AtomRings())
    if len(rings) != 1:
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.OUTSIDE_PROFILE,
            reason="The trigger requires one physical carbon monocycle.",
        )
    ring_atoms = frozenset(rings[0])
    if (len(ring_atoms) < 3
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring_atoms)
            or any(mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1
                   for i in ring_atoms
                   for j in (neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(i).GetNeighbors())
                   if j in ring_atoms)):
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.OUTSIDE_PROFILE,
            ring_atoms=ring_atoms,
            reason="Only saturated all-carbon monocycle parents enter this implicit-feature trigger.",
        )
    inventory = _source_inventory_evidence(mol, Chem)
    identities = {_group_identity(group) for group in groups}
    if (not inventory.supported
            or inventory.source_status not in {"selected", "tie"}
            or identities != inventory.source_ids or identities != inventory.principal_ids
            or any(group.center in ring_atoms or group.owner not in ring_atoms
                   for group in groups)):
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.OUTSIDE_PROFILE,
            ring_atoms=ring_atoms,
            reason="The complete source-selected suffix set is not attached to this ring.",
        )

    features = _potential_feature_census(mol, Chem)
    if features is None:
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.UNRESOLVED,
            ring_atoms=ring_atoms,
            reason="Fresh map-free potential-stereo census failed for a qualified suffix ring.",
        )

    ring_features = tuple(feature for feature in features if feature[1] in ring_atoms)
    if not features:
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.NO_TRIGGER,
            ring_atoms=ring_atoms,
            feature_census=features,
            reason="The qualified ring has an empty potential-stereo census.",
        )
    if (not ring_features
            or len(ring_features) != len(features)
            or any(feature[0] != "Atom_Tetrahedral"
                   or mol.GetAtomWithIdx(feature[1]).GetAtomicNum() != 6
                   or feature[2] != "Unspecified"
                   or feature[3] not in {"NoValue", "None"}
                   for feature in ring_features)):
        return PotentialStereoTrigger(
            PotentialStereoTriggerStatus.UNRESOLVED,
            ring_atoms=ring_atoms,
            feature_census=features,
            triggering_atoms=tuple(sorted({feature[1] for feature in ring_features})),
            reason="A qualified suffix ring has a non-parent, non-tetrahedral, or otherwise unsupported potential feature.",
        )
    return PotentialStereoTrigger(
        PotentialStereoTriggerStatus.CANDIDATE,
        ring_atoms=ring_atoms,
        feature_census=features,
        triggering_atoms=tuple(sorted(feature[1] for feature in ring_features)),
        reason="Unspecified parent-carbon tetrahedral units require the mapped P-91 proof and Xi writer.",
    )


def prove_monocycle_parent_annotations(mol, Chem, *, parent_atoms, parent_locants,
                                       principal_groups, principal_class):
    """Validate the source-closed C-13/ordinary-RS/Xi ring annotation profile."""
    parent_atoms = frozenset(int(idx) for idx in parent_atoms)
    parent_locants = tuple(sorted((int(idx), str(locant))
                                  for idx, locant in dict(parent_locants).items()))
    principal_groups = tuple(principal_groups)
    principal_identity = tuple(sorted(_group_identity(group) for group in principal_groups))
    if not parent_atoms or {idx for idx, _ in parent_locants} != set(parent_atoms):
        return _outside("Parent atom locants do not cover the complete source parent.")
    if not _canonical_consecutive_ring_locants(mol, parent_atoms, parent_locants):
        return _outside("Parent ring locants are not a unique consecutive cycle numbering.")
    if principal_class not in {"acid", "amide", "nitrile"} or not principal_groups:
        return _outside("Only the source-closed acid, amide, and nitrile ring classes are admitted.")

    # Build on a fresh, same-order copy on the first call or after any native
    # graph/property edit. Only immutable identities are shared across maps;
    # stale Python inventory caches cannot authorize source evidence.
    inventory = _source_inventory_evidence(mol, Chem)
    if not inventory.supported:
        return _outside("The original functional-group inventory is incomplete.")
    if any(_group_identity(group) not in inventory.group_ids for group in principal_groups):
        return _outside("A selected suffix group is not owned by the original inventory.")
    if (inventory.source_status not in {"selected", "tie"}
            or inventory.source_ids != frozenset(principal_identity)
            or inventory.principal_ids != frozenset(principal_identity)):
        return _outside("The supplied suffix groups are not the complete source-selected principal set.")
    if any(group.kind != principal_class for group in principal_groups):
        return _outside("Selected suffix groups do not share the declared source class.")

    # This proof is intentionally limited to one physical all-carbon ring
    # whose suffix core and every N-branch remain unlabelled and unannotated.
    ring_ids = frozenset(parent_atoms)
    physical_rings = tuple(frozenset(row) for row in mol.GetRingInfo().AtomRings())
    if len(physical_rings) != 1 or physical_rings[0] != ring_ids:
        return _outside("The selected parent is not the unique physical monocycle.")
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in parent_atoms):
        return _outside("The selected monocycle is not all carbon.")
    suffix_core = tuple(sorted(
        (identity, tuple(sorted(int(i) for i in group.atoms)))
        for identity, group in ((_group_identity(g), g) for g in principal_groups)
    ))
    suffix_atoms = set()
    n_branch_ownership = []
    for group in principal_groups:
        suffix_atoms.update(int(i) for i in group.atoms)
        suffix_atoms.add(int(group.center))
        if group.ligand >= 0:
            # The suffix-core recognizer owns the N/O attachment atom, while
            # the mapped suffix writer also owns its detached substituent
            # branches. Include the complete component beyond that bond so a
            # label cannot escape the source profile through an N branch.
            stack = [int(group.ligand)]
            branch_atoms = set()
            while stack:
                idx = stack.pop()
                if idx in branch_atoms:
                    continue
                branch_atoms.add(idx)
                stack.extend(
                    neighbor.GetIdx()
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if neighbor.GetIdx() != int(group.center)
                    and neighbor.GetIdx() not in branch_atoms
                )
            suffix_atoms.update(branch_atoms)
            if group.kind == "amide":
                n_branch_ownership.append((
                    _group_identity(group), tuple(sorted(branch_atoms)),
                ))
    if suffix_atoms & parent_atoms:
        return _outside("Suffix-core ownership overlaps the selected parent ring.")
    if any(mol.GetAtomWithIdx(idx).GetIsotope()
           or mol.GetAtomWithIdx(idx).GetChiralTag()
           for idx in suffix_atoms):
        return _outside("Suffix-core and N-branch atoms must be unlabelled and unannotated.")

    isotope_ownership = []
    for atom in mol.GetAtoms():
        mass = int(atom.GetIsotope())
        if not mass:
            continue
        idx = atom.GetIdx()
        if atom.GetAtomicNum() != 6 or mass != 13 or idx not in parent_atoms:
            return _outside("Only parent-ring carbon-13 isotope ownership is admitted.")
        isotope_ownership.append((idx, atom.GetAtomicNum(), mass, "parent_carbon"))

    if mol.GetStereoGroups():
        return _outside("Enhanced or grouped stereochemistry is outside this profile.")
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in mol.GetBonds()):
        return _outside("Bond stereochemistry is outside this profile.")

    original_census = _potential_feature_census(mol, Chem)
    if original_census is None:
        return _blocked("Original potential-stereo census failed.")
    cleared_census = _isotope_cleared_feature_census(mol, Chem)
    if cleared_census is None:
        return _blocked("Isotope-cleared potential-stereo census failed.")
    if original_census != cleared_census:
        return _blocked(
            "Isotope substitution creates or removes a stereogenic unit; P-82.5.1 numbering remains unresolved.",
            graph_signature=_graph_signature(mol), principal_identity=principal_identity,
            principal_class=principal_class, parent_atoms=parent_atoms,
            parent_locants=parent_locants, suffix_core_ownership=suffix_core,
            isotope_ownership=tuple(sorted(isotope_ownership)),
            original_feature_census=original_census,
            isotope_cleared_feature_census=cleared_census,
        )

    try:
        assigned = dict(_assigned_parent_descriptors(mol, Chem))
    except Exception:
        return _blocked("Modern assigned-configuration evidence is unavailable.")

    features = []
    potential_atom_ids = set()
    unknown_atom_ids = tuple(sorted(
        atom_id for feature_type, atom_id, specified, descriptor in original_census
        if feature_type == "Atom_Tetrahedral"
        and mol.GetAtomWithIdx(atom_id).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
        and specified == "Unspecified" and descriptor in {"NoValue", "None"}
    ))
    # The descriptor class of one unknown unit can depend on other unknown
    # units in its ligands. Exhaustively classify the small closed ring family;
    # larger unresolved sets remain outside this proof rather than being
    # treated as capital R/S from a single arbitrary completion.
    if len(unknown_atom_ids) > 8:
        return _outside("Too many mutually unresolved tetrahedral features for a closed descriptor-class proof.")
    for feature_type, atom_id, specified, descriptor in original_census:
        if feature_type != "Atom_Tetrahedral":
            return _outside("Only tetrahedral parent-carbon stereogenic units are admitted.")
        if atom_id not in parent_atoms or mol.GetAtomWithIdx(atom_id).GetAtomicNum() != 6:
            return _outside("Every stereogenic unit must be owned by a parent-ring carbon.")
        potential_atom_ids.add(atom_id)
        atom = mol.GetAtomWithIdx(atom_id)
        tag = atom.GetChiralTag()
        locant = dict(parent_locants).get(atom_id)
        if not locant:
            return _blocked("A parent stereogenic unit has no mapped ring locant.")
        if tag != Chem.ChiralType.CHI_UNSPECIFIED:
            label = assigned.get(atom_id)
            if label not in {"R", "S"}:
                return _outside("Assigned pseudoasymmetric or unsupported configuration is outside this profile.")
            features.append(ParentStereoFeature(
                atom_id, feature_type, "assigned", "capital_RS", label, locant,
            ))
        else:
            if specified != "Unspecified" or descriptor not in {"NoValue", "None"}:
                return _outside("Potential stereogenic configuration state is not an ordinary unknown.")
            if (not _unknown_ligands_are_constitutionally_distinct(mol, Chem, atom_id)
                    or not _unknown_is_ordinary_capital_rs(
                        mol, Chem, atom_id, unknown_atom_ids)):
                return _blocked("Unknown stereogenic unit is not proved to have the ordinary capital R/S class.")
            features.append(ParentStereoFeature(
                atom_id, feature_type, "unknown_ordinary", "capital_RS", "Ξ", locant,
            ))

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and idx not in potential_atom_ids:
            return _outside("An assigned stereochemical annotation lacks a parent feature owner.")

    return ParentAnnotationProof(
        ParentAnnotationStatus.PROVED,
        _graph_signature(mol), principal_identity, principal_class, parent_atoms,
        parent_locants, suffix_core, tuple(sorted(isotope_ownership)),
        original_census, cleared_census,
        tuple(sorted(features, key=lambda row: int(row.locant))),
        ("P-91.2.1.2.1", "P-91.3", "P-92.1.1", "P-92.3",
         "P-82.4", "P-82.5.1", "P-82.5.2"),
        "Parent isotope and ordinary assigned/unknown stereogenic features have complete original-atom ownership.",
        tuple(sorted(n_branch_ownership)),
    )


def validate_monocycle_parent_annotation_proof(mol, Chem, proof, *,
                                                parent_locants, principal_groups):
    """Rebuild and compare proof on the untouched original graph before writing."""
    if not isinstance(proof, ParentAnnotationProof):
        return False
    if proof.status is not ParentAnnotationStatus.PROVED:
        return False
    fresh = prove_monocycle_parent_annotations(
        mol, Chem, parent_atoms=proof.parent_atoms, parent_locants=parent_locants,
        principal_groups=principal_groups, principal_class=proof.principal_class,
    )
    return fresh == proof


def _canonical_consecutive_ring_locants(mol, parent_atoms, parent_locants):
    locant_by_atom = dict(parent_locants)
    size = len(parent_atoms)
    expected = {str(value) for value in range(1, size + 1)}
    if (set(locant_by_atom) != set(parent_atoms)
            or set(locant_by_atom.values()) != expected):
        return False
    atom_by_locant = {locant: atom for atom, locant in locant_by_atom.items()}
    if len(atom_by_locant) != size:
        return False
    for value in range(1, size + 1):
        atom_id = atom_by_locant[str(value)]
        previous = str(size if value == 1 else value - 1)
        following = str(1 if value == size else value + 1)
        ring_neighbors = {
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors()
            if neighbor.GetIdx() in parent_atoms
        }
        if ring_neighbors != {atom_by_locant[previous], atom_by_locant[following]}:
            return False
    return True


def _unknown_is_ordinary_capital_rs(mol, Chem, atom_id, unknown_atom_ids):
    """Classify an unassigned unit under every assignment of other unknowns.

    Ordinary tetrahedral chirality maps to the capital R/S descriptor pair;
    pseudoasymmetry maps to lower-case r/s. No configuration of the original
    input is inferred by assigning both temporary tags.
    """
    other_ids = tuple(idx for idx in unknown_atom_ids if idx != atom_id)
    if len(other_ids) > 8:
        return False
    ids = tuple(sorted({atom_id, *other_ids}))
    return dict(_ordinary_rs_by_atom(mol, Chem, ids))[atom_id]


def _ordinary_rs_by_atom(mol, Chem, unknown_atom_ids):
    """Share each complete assignment among all unknown atoms and numberings.

    A CIP assignment labels the whole graph. The old per-target traversal
    repeated the same 2**k assignments k times for every parent orientation.
    Retain its pair test for every fixed assignment of the other atoms, but
    compute each full assignment at most once. Only immutable classifications
    survive the computation; neither scratch molecules nor locants are cached.
    """
    from .stereochemistry import assigned_atom_stereochemistry

    def compute():
        tags = (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        assignments = {}

        def labels(mask):
            if mask not in assignments:
                scratch = Chem.Mol(mol)
                _clear_atom_maps(scratch)
                for index, atom_id in enumerate(unknown_atom_ids):
                    scratch.GetAtomWithIdx(atom_id).SetChiralTag(tags[(mask >> index) & 1])
                try:
                    assigned = dict(assigned_atom_stereochemistry(
                        scratch, Chem, atom_order="idx", include_pseudo=True,
                    ))
                    assignments[mask] = tuple(assigned.get(idx) for idx in unknown_atom_ids)
                except Exception:
                    assignments[mask] = None
            return assignments[mask]

        def ordinary(position):
            others = tuple(index for index in range(len(unknown_atom_ids))
                           if index != position)
            for bits in product((0, 1), repeat=len(others)):
                mask = sum(bit << index for bit, index in zip(bits, others))
                left = labels(mask)
                if left is None or left[position] not in {"R", "S"}:
                    return False
                right = labels(mask | (1 << position))
                if right is None or {left[position], right[position]} != {"R", "S"}:
                    return False
            return True

        return tuple((atom_id, ordinary(index))
                     for index, atom_id in enumerate(unknown_atom_ids))

    return exact_graph_computation(
        mol, Chem, "monocycle_ordinary_rs_classes",
        (Chem, assigned_atom_stereochemistry, unknown_atom_ids), compute,
    )


def _unknown_ligands_are_constitutionally_distinct(mol, Chem, atom_id):
    """Require distinct atom-mapped rooted ligand symmetry classes.

    This conservative graph check rejects a candidate whose stereo class could
    depend on enantiomorphic/equivalent ligands. A unique temporary map anchors
    the center while isotope masses, atom maps, tetra tags and bond stereo are
    removed on a scratch copy. RDKit's symmetry classes then compare ligands
    using constitutional graph identity only.
    """
    return exact_graph_computation(
        mol, Chem, "monocycle_constitutional_ligands",
        (Chem, getattr(Chem, "CanonicalRankAtoms", None),
         getattr(Chem, "SanitizeMol", None), atom_id),
        lambda: _uncached_constitutional_ligands(mol, Chem, atom_id),
    )


def _uncached_constitutional_ligands(mol, Chem, atom_id):
    scratch = Chem.Mol(mol)
    _clear_atom_maps(scratch)
    for atom in scratch.GetAtoms():
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in scratch.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    scratch.GetAtomWithIdx(atom_id).SetAtomMapNum(9001)
    try:
        Chem.SanitizeMol(scratch)
        ranks = tuple(Chem.CanonicalRankAtoms(
            scratch, breakTies=False, includeChirality=False,
            includeIsotopes=False, includeAtomMaps=True,
            includeChiralPresence=False,
        ))
    except Exception:
        return False
    anchor_class = ranks[atom_id]
    if any(rank == anchor_class for idx, rank in enumerate(ranks)
           if idx != atom_id):
        return False
    neighbors = tuple(neighbor.GetIdx()
                      for neighbor in scratch.GetAtomWithIdx(atom_id).GetNeighbors())
    if len(neighbors) not in {3, 4}:
        return False
    classes = tuple(ranks[idx] for idx in neighbors)
    return len(set(classes)) == len(classes)


def _potential_feature_census(mol, Chem):
    return exact_graph_computation(
        mol, Chem, "monocycle_feature_census",
        (Chem, getattr(Chem, "FindPotentialStereo", None)),
        lambda: _uncached_feature_census(mol, Chem),
    )


def _uncached_feature_census(mol, Chem):
    scratch = Chem.Mol(mol)
    _clear_atom_maps(scratch)
    try:
        return tuple(sorted((str(feature.type).split(".")[-1],
                             int(feature.centeredOn),
                             str(feature.specified).split(".")[-1],
                             str(feature.descriptor).split(".")[-1])
                            for feature in Chem.FindPotentialStereo(scratch)))
    except Exception:
        return None


def _isotope_cleared_feature_census(mol, Chem):
    def compute():
        scratch = Chem.Mol(mol)
        for atom in scratch.GetAtoms():
            atom.SetIsotope(0)
        _clear_atom_maps(scratch)
        Chem.SanitizeMol(scratch)
        return _potential_feature_census(scratch, Chem)

    return exact_graph_computation(
        mol, Chem, "monocycle_isotope_cleared_census",
        (Chem, getattr(Chem, "FindPotentialStereo", None),
         getattr(Chem, "SanitizeMol", None)), compute,
    )


def _assigned_parent_descriptors(mol, Chem):
    from .stereochemistry import assigned_atom_stereochemistry

    def compute():
        scratch = Chem.Mol(mol)
        _clear_atom_maps(scratch)
        return tuple(assigned_atom_stereochemistry(
            scratch, Chem, atom_order="idx", include_pseudo=True,
        ))

    return exact_graph_computation(
        mol, Chem, "monocycle_assigned_descriptors",
        (Chem, assigned_atom_stereochemistry), compute,
    )


def _source_inventory_evidence(mol, Chem):
    from .extended_inventory import selection_inventory

    def compute():
        inventory = selection_inventory(Chem.Mol(mol))
        if not inventory.supported:
            return _SourceInventoryEvidence(False)
        frontier = getattr(inventory, "source_frontier", None)
        return _SourceInventoryEvidence(
            True,
            frozenset(_group_identity(group) for group in inventory.groups),
            getattr(frontier, "status", None),
            frozenset(_group_identity(group) for group in getattr(frontier, "groups", ())),
            frozenset(_group_identity(group) for group in inventory.principal),
        )

    return exact_graph_computation(
        mol, Chem, "monocycle_source_inventory",
        (Chem, selection_inventory), compute,
    )


def _clear_atom_maps(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)


def _graph_signature(mol):
    atoms = tuple((atom.GetIdx(), atom.GetAtomicNum(), atom.GetIsotope(),
                   atom.GetFormalCharge(), atom.GetNumRadicalElectrons(),
                   int(atom.GetChiralTag()), atom.GetNumExplicitHs(),
                   atom.GetNoImplicit(), atom.GetIsAromatic())
                  for atom in mol.GetAtoms())
    bonds = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                          str(bond.GetBondType()), bond.GetIsAromatic(),
                          int(bond.GetStereo()), int(bond.GetBondDir()))
                         for bond in mol.GetBonds()))
    return atoms, bonds


def _group_identity(group):
    return (str(group.kind), int(group.center), tuple(int(i) for i in group.atoms),
            int(group.owner), int(group.ligand))


def _locant_sort_key(value):
    token = str(value)
    return (0, int(token)) if token.isdigit() else (1, token)


def _outside(reason):
    return ParentAnnotationProof(ParentAnnotationStatus.OUT_OF_SCOPE, reason=reason)


def _blocked(reason, **fields):
    return ParentAnnotationProof(ParentAnnotationStatus.BLOCKED, reason=reason, **fields)
