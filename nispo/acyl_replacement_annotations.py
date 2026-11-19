"""Original-graph P-82/P-9 decoration for fixed acyl-replacement parents.

This adapter accepts the unannotated constitutional proof from the acyl
replacement producer and decorates its already fixed parent on the original
annotated graph. It does not choose the parent path or assign suffix locants.
"""
from __future__ import annotations

from dataclasses import dataclass

from .naming_methods import MethodDecision


_ANNOTATION_RULES = "/P-82.2.1/P-82.4/P-82.5.1/P-9.3.4.1"


@dataclass(frozen=True, slots=True)
class AnnotatedAcylReplacementDecision(MethodDecision):
    """Complete emitted decision with original-graph annotation evidence."""

    structural_proof: object | None = None
    decoration: object | None = None
    isotope_coverage: object | None = None
    equivalent_atom_locants: tuple = ()
    fixed_suffix_locant: str = ""
    original_heavy_ownership: frozenset[int] = frozenset()
    isotope_hydrogen_bindings: tuple[tuple[int, int, str], ...] = ()

@dataclass(frozen=True, slots=True)
class AcylReplacementAnnotation:
    """A decorated fixed-name decision plus the same-writer map evidence."""

    decision: AnnotatedAcylReplacementDecision
    structural_proof: object
    decoration: object
    parent_atoms: frozenset[int]
    suffix_atoms: frozenset[int]


def decorate_acyl_replacement_proof(mol, Chem, proof):
    """Decorate only parent isotope and assigned tetrahedral stereo roles.

    ``proof`` must be an ``AcylReplacementProof`` obtained from the
    isotope/stereo-cleared constitutional graph with unchanged heavy-atom
    indices. Isotope-labeled atoms must be on its fixed parent path, and
    isotope-labeled explicit hydrogen atoms must be bonded directly to that
    path. Suffix O/N/H isotope roles, bond stereochemistry, and stereo outside
    the fixed parent remain unsupported and return ``None``.
    """
    from .acyl_replacement_methods import (
        AcylReplacementProof, _parent_text, decorate_replacement_prefixes, replacement_leaf_prefix,
    )
    from .component_annotation_evidence import decorate_component_with_evidence
    from .component_mapping import MappedComponent, hydrogen_bindings
    from .isotope_component_coverage import validate_isotope_component_coverage

    if not isinstance(proof, AcylReplacementProof) or mol.GetStereoGroups():
        return None
    path = tuple(proof.parent_path)
    if (not path or len(set(path)) != len(path)
            or any(not isinstance(i, int) or isinstance(i, bool)
                   or not 0 <= i < mol.GetNumAtoms() for i in path)):
        return None
    parent_atoms = frozenset(path)
    decision = proof.decision
    if (decision is None or decision.method != "skeletal-replacement"
            or tuple(dict(decision.atom_locants)) != tuple(path)):
        return None
    roles = tuple(decision.component_roles)
    prefixes = proof.prefixes
    prefix_atoms = frozenset(idx for idx, _, _ in prefixes)
    if (len(roles) != (3 if prefixes else 2) or roles[0][0] != "replacement_parent"
            or roles[1][0] != "principal_suffix"
            or frozenset(roles[0][1]) != parent_atoms
            or len(roles[0][1]) != len(parent_atoms)):
        return None
    if prefixes:
        if (roles[2][0] != 'substituent_prefixes' or roles[2][1] != prefix_atoms
                or any(row[1] not in parent_atoms or replacement_leaf_prefix(mol, row[0]) != row
                       for row in prefixes)):
            return None
    suffix_atoms = frozenset(roles[1][1])
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if (parent_atoms & suffix_atoms or prefix_atoms & (parent_atoms | suffix_atoms)
            or parent_atoms | suffix_atoms | prefix_atoms != heavy
            or not suffix_atoms or proof.decision.atoms != heavy):
        return None
    if len(decision.principal_centers) != 1:
        return None
    center = next(iter(decision.principal_centers))
    if center not in parent_atoms or path.index(center) + 1 != int(proof.suffix_locant):
        return None
    if proof.suffix_kind not in {"acid", "amide", "nitrile"}:
        return None
    if (mol.GetAtomWithIdx(center).GetAtomicNum() != 6
            or any(mol.GetBondBetweenAtoms(a, b) is None
                   or mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1.0
                   for a, b in zip(path, path[1:]))):
        return None
    expected_suffix_elements = {
        "acid": (8, 8), "amide": (7, 8), "nitrile": (7,),
    }[proof.suffix_kind]
    actual_suffix_elements = tuple(sorted(mol.GetAtomWithIdx(i).GetAtomicNum()
                                           for i in suffix_atoms))
    if actual_suffix_elements != tuple(sorted(expected_suffix_elements)):
        return None
    if any(mol.GetBondBetweenAtoms(center, i) is None for i in suffix_atoms):
        return None
    if proof.suffix_kind == "nitrile":
        if mol.GetBondBetweenAtoms(center, next(iter(suffix_atoms))).GetBondTypeAsDouble() != 3.0:
            return None
    else:
        double_oxygens = [i for i in suffix_atoms
                          if mol.GetAtomWithIdx(i).GetAtomicNum() == 8
                          and mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() == 2.0]
        if len(double_oxygens) != 1:
            return None
        single_atoms = suffix_atoms - set(double_oxygens)
        if (len(single_atoms) != 1
                or any(mol.GetBondBetweenAtoms(center, i).GetBondTypeAsDouble() != 1.0
                       for i in single_atoms)):
            return None

    # Every label is either a parent atom or an explicit isotope H attached
    # directly to a mapped parent atom. Suffix labels require their own typed
    # word-level role and are deliberately not guessed here.
    isotope_hydrogens = set()
    for atom in mol.GetAtoms():
        if not atom.GetIsotope():
            continue
        idx = atom.GetIdx()
        if idx in parent_atoms and atom.GetAtomicNum() > 1:
            continue
        if atom.GetAtomicNum() != 1:
            return None
        hosts = [n.GetIdx() for n in atom.GetNeighbors()
                 if n.GetAtomicNum() > 1]
        if len(hosts) != 1 or hosts[0] not in parent_atoms:
            return None
        isotope_hydrogens.add(idx)

    # This writer contract carries tetrahedral parent CIP only. Any assigned
    # or potential stereochemistry outside the mapped parent is a decline.
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE
           or b.GetBondDir() != Chem.BondDir.NONE for b in mol.GetBonds()):
        return None
    if any(int(atom.GetChiralTag()) and atom.GetIdx() not in parent_atoms
           for atom in mol.GetAtoms()):
        return None

    parent = _parent_text(mol, path)
    if not parent:
        return None
    endings = {"acid": "oic acid", "amide": "amide", "nitrile": "nitrile"}
    unmodified_name = f"{parent[:-1]}-{proof.suffix_locant}-{endings[proof.suffix_kind]}"
    heavy_locants = dict(decision.atom_locants)
    if decision.name != decorate_replacement_prefixes(unmodified_name, heavy_locants, prefixes):
        return None

    mapped_locants = hydrogen_bindings(mol, heavy_locants)
    component_atoms = parent_atoms | frozenset(isotope_hydrogens)
    component_locants = tuple((idx, mapped_locants[idx])
                              for idx in (*path, *sorted(isotope_hydrogens)))
    component = MappedComponent(
        parent=parent,
        atom_locants=component_locants,
        atoms=component_atoms,
        prefixes=tuple((heavy_locants[owner], word) for _, owner, word in prefixes),
        embedded_atoms=prefix_atoms,
        systematic_parent=True,
    )
    decoration = decorate_component_with_evidence(
        mol, Chem, component, component_id="acyl-replacement:fixed-parent")
    if decoration is None or decoration.evidence_scope != component_atoms:
        return None
    expected_isotope_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsotope()}
    claimed_isotope_atoms = {row.atom_id for row in decoration.isotope_record.isotope_atoms}
    if claimed_isotope_atoms != expected_isotope_atoms:
        return None
    coverage = validate_isotope_component_coverage(mol, (decoration.isotope_record,))
    if not coverage.complete:
        return None

    # The typed writer renders the fixed parent, isotope descriptors, and
    # stereo at the source-defined positions. The acyl suffix is transferred
    # only after confirming the decorated text ends in this exact parent.
    if not decoration.text.endswith(parent):
        return None
    front = decoration.text[:-len(parent)]
    rendered = f"{front}{parent[:-1]}-{proof.suffix_locant}-{endings[proof.suffix_kind]}"
    equivalent_maps = []
    for equivalent_path in proof.equivalent_paths or (path,):
        equivalent_path = tuple(equivalent_path)
        if set(equivalent_path) != parent_atoms or len(equivalent_path) != len(path):
            return None
        equivalent_maps.append(tuple((idx, str(position))
                                     for position, idx in enumerate(equivalent_path, 1)))
    equivalent_maps = tuple(dict.fromkeys(equivalent_maps))
    isotope_hydrogen_bindings = tuple(
        (idx, next(iter(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                        if n.GetAtomicNum() > 1)), mapped_locants[idx])
        for idx in sorted(isotope_hydrogens)
    )
    if any(host not in parent_atoms for _hydrogen, host, _locant in isotope_hydrogen_bindings):
        return None
    decorated_decision = AnnotatedAcylReplacementDecision(
        name=rendered,
        method=decision.method,
        principal_centers=decision.principal_centers,
        atoms=decision.atoms,
        rule=decision.rule + _ANNOTATION_RULES,
        atom_locants=decision.atom_locants,
        heterounits=decision.heterounits,
        component_roles=decision.component_roles,
        functional_parent_frontier=decision.functional_parent_frontier,
        structural_proof=proof,
        decoration=decoration,
        isotope_coverage=coverage,
        equivalent_atom_locants=equivalent_maps,
        fixed_suffix_locant=str(proof.suffix_locant),
        original_heavy_ownership=heavy,
        isotope_hydrogen_bindings=isotope_hydrogen_bindings,
    )
    return AcylReplacementAnnotation(
        decision=decorated_decision,
        structural_proof=proof,
        decoration=decoration,
        parent_atoms=parent_atoms,
        suffix_atoms=suffix_atoms,
    )
