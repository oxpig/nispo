"""Original-graph annotation admission for the ordinary suffix-site frontier."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrincipalSiteAnnotationStatus(str, Enum):
    PROVED = "proved"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class PrincipalSiteAnnotationProof:
    """Original-atom ownership and constitutional stereo identity evidence."""

    status: PrincipalSiteAnnotationStatus
    isotope_atoms: tuple[int, ...] = ()
    isotope_roles: tuple[tuple[int, int, int], ...] = ()
    assigned_stereo_atoms: tuple[tuple[int, str], ...] = ()
    potential_feature_ids: tuple[tuple[str, int], ...] = ()
    isotope_cleared_feature_ids: tuple[tuple[str, int], ...] = ()
    maximum_parent_atom_sets: tuple[frozenset[int], ...] = ()
    source_rule_order: tuple[str, ...] = ()
    reason: str = ""
    assigned_stereo_bonds: tuple = ()


def prove_parent_annotations(mol, Chem, candidates):
    """Reuse the writer's complete chain census on constitutional maxima.

    The older one-C13/one-R/S adapter remains available for partially covered
    site families. Complete-coverage families can carry the same multiple
    parent labels, assigned R/S and E/Z as the existing mapped chain writer.
    Carbon/deuterium/tritium labels must be parent-owned on every maximum;
    isotope-created stereo, unspecified features and enhanced groups decline.
    """
    candidates = tuple(candidates)
    if not candidates or any(candidate.omitted_sites for candidate in candidates):
        return prove_parent_carbon13_stereo(mol, Chem, candidates)
    if mol.GetStereoGroups():
        return _annotation_outside("Enhanced or grouped stereochemistry is outside this admission.")
    from .complete_functional_parent import _supports_annotated_chain_frontier
    from .extended_inventory import selection_inventory
    from .stereochemistry import assigned_atom_stereochemistry, _indexed_bond_stereochemistry

    inventory = selection_inventory(mol)
    groups = tuple(inventory.principal)
    skeletons = tuple((candidate.parent_atoms, False, False) for candidate in candidates)
    if not _supports_annotated_chain_frontier(mol, Chem, groups, skeletons):
        return _annotation_outside("The mapped chain writer cannot own every isotope/stereo feature on every maximum.")
    isotope_atoms = tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope())
    isotope_roles = tuple((idx, mol.GetAtomWithIdx(idx).GetAtomicNum(),
                           mol.GetAtomWithIdx(idx).GetIsotope()) for idx in isotope_atoms)
    assigned = tuple(sorted(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True,
    )))
    original_features = _potential_feature_ids(mol, Chem)
    cleared = Chem.Mol(mol)
    for atom in cleared.GetAtoms():
        atom.SetIsotope(0)
    _clear_provenance_maps(cleared)
    Chem.SanitizeMol(cleared)
    cleared_features = _potential_feature_ids(cleared, Chem)
    if original_features != cleared_features:
        return _annotation_outside("Clearing isotope masses changes potential stereogenic-feature identities.")
    return PrincipalSiteAnnotationProof(
        PrincipalSiteAnnotationStatus.PROVED, isotope_atoms, isotope_roles,
        assigned, original_features, cleared_features,
        tuple(frozenset(candidate.parent_atoms) for candidate in candidates),
        ("P-44.4.1.8", "P-44.4.1.10", "P-44.4.1.11", "P-44.4.1.12", "P-82.5.1", "P-92"),
        "Every constitutional maximum retains the mapped chain writer's complete parent isotope/stereo ownership.",
        tuple(_indexed_bond_stereochemistry(mol, Chem)),
    )


def prove_parent_carbon13_stereo(mol, Chem, candidates):
    """Prove the sole admitted annotation profile on every constitutional max.

    The accepted profile is exactly one parent C-13 atom plus one explicitly
    assigned ordinary R/S carbon center. Stereo feature identity is compared
    on scratch copies with isotope masses cleared; the CIP letter is allowed
    to change because isotope mass can change ligand priority without creating
    a stereogenic feature.
    """
    from .stereochemistry import assigned_atom_stereochemistry

    candidates = tuple(candidates)
    if not candidates:
        return _annotation_outside("No constitutional maximum is available for annotation proof.")
    isotope_atoms = tuple(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope())
    if len(isotope_atoms) != 1:
        return _annotation_outside("Only one explicitly labelled parent carbon is admitted.")
    isotope_idx = isotope_atoms[0]
    isotope_atom = mol.GetAtomWithIdx(isotope_idx)
    if isotope_atom.GetAtomicNum() != 6 or isotope_atom.GetIsotope() != 13:
        return _annotation_outside("Only one parent carbon-13 label is admitted.")
    if mol.GetStereoGroups():
        return _annotation_outside("Enhanced or grouped stereochemistry is outside this admission.")

    tagged = tuple(atom.GetIdx() for atom in mol.GetAtoms()
                   if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    if len(tagged) != 1:
        return _annotation_outside("Exactly one assigned atom stereocenter is admitted.")
    stereo_idx = tagged[0]
    if mol.GetAtomWithIdx(stereo_idx).GetAtomicNum() != 6:
        return _annotation_outside("Only an assigned carbon R/S stereocenter is admitted.")

    # This annotation extension is deliberately saturated. The earlier
    # unannotated Job328 frontier remains responsible for unsaturated trees.
    if any(bond.GetBondType() != Chem.BondType.SINGLE
           for bond in mol.GetBonds()
           if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 6
           and mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 6):
        return _annotation_outside("Annotated unsaturated carbon trees remain outside this admission.")
    if any(int(bond.GetStereo()) or int(bond.GetBondDir()) for bond in mol.GetBonds()):
        return _annotation_outside("Assigned bond stereo or directional bonds are outside this admission.")

    parent_sets = tuple(frozenset(candidate.parent_atoms) for candidate in candidates)
    if any(isotope_idx not in parent for parent in parent_sets):
        return _annotation_outside("The C-13 atom must be a parent atom on every constitutional maximum.")
    if any(stereo_idx not in parent for parent in parent_sets):
        return _annotation_outside("The assigned stereocenter must be on every constitutional maximum.")

    original_features = _potential_feature_ids(mol, Chem)
    cleared = Chem.Mol(mol)
    for atom in cleared.GetAtoms():
        atom.SetIsotope(0)
    _clear_provenance_maps(cleared)
    Chem.SanitizeMol(cleared)
    cleared_features = _potential_feature_ids(cleared, Chem)
    if original_features != cleared_features:
        return _annotation_outside(
            "Clearing isotope masses changes potential stereogenic-feature identities."
        )

    assigned = tuple(sorted(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True,
    )))
    if assigned != ((stereo_idx, "R"),) and assigned != ((stereo_idx, "S"),):
        return _annotation_outside("The assigned carbon center is not one supported ordinary R/S feature.")
    expected_feature = (("Atom_Tetrahedral", stereo_idx),)
    if original_features != expected_feature:
        return _annotation_outside(
            "The molecule has additional or unsupported potential stereogenic features."
        )

    return PrincipalSiteAnnotationProof(
        PrincipalSiteAnnotationStatus.PROVED,
        isotope_atoms,
        ((isotope_idx, 6, 13),),
        assigned,
        original_features,
        cleared_features,
        parent_sets,
        ("P-44.4.1.8", "P-44.4.1.11", "P-44.4.1.12", "P-82.5.1", "P-92"),
        "One parent C-13 and one constitutionally existing assigned carbon R/S feature are mapped on every constitutional maximum.",
    )


def _potential_feature_ids(mol, Chem):
    """Find potential stereo on a detached, map-free copy of the input graph."""
    scratch = Chem.Mol(mol)
    _clear_provenance_maps(scratch)
    return tuple(sorted(
        (str(feature.type).split(".")[-1], int(feature.centeredOn))
        for feature in Chem.FindPotentialStereo(scratch)
    ))


def _clear_provenance_maps(mol):
    # Maps identify source atoms for proposal ownership; they do not define
    # ligand identity and must not affect potential-stereo discovery.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)


def _annotation_outside(reason):
    return PrincipalSiteAnnotationProof(
        PrincipalSiteAnnotationStatus.OUT_OF_SCOPE,
        reason=reason,
    )
