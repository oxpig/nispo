"""Typed isotope roles for fixed P-51 acyl-replacement suffixes.

This module records original-graph ownership for isotope labels outside the
numbered replacement-parent path.  It is intentionally a sidecar: selection,
parent decoration, and complete-name insertion remain with the caller.

The initial closed grammar is narrow.  It recognizes hydroxyl-bound isotope H
on a terminal carboxylic acid and isotope N / N-bound isotope H on a primary
amide.  Acid oxygen-only forms and nitrile nitrogen labels remain explicitly
pending because their suffix-local insertion syntax is not closed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .acyl_replacement_annotations import AnnotatedAcylReplacementDecision


RULES = (
    "P-82.2.1", "P-82.2.4", "P-82.2.5", "P-82.3.1", "P-82.3.2",
    "P-82.6.1", "P-82.6.2",
    "P-82.6.3.1", "P-82.6.4", "P-65.1.2.1", "P-66.1.1.1.3",
    "P-66.1.1.3.1.1", "P-66.5.1.1.3",
)


@dataclass(frozen=True, slots=True)
class SuffixIsotopePart:
    """One primary nuclide owner and its name insertion target."""

    symbol: str
    mass: int
    locant: str
    atom_id: int
    suffix_role: str
    insertion_target: str
    count: int = 1


@dataclass(frozen=True, slots=True)
class SuffixIsotopeEvidence:
    """Validated role evidence for one mapped terminal suffix.

    ``parts`` own isotope-modified original atoms exactly once.  A hydrogen
    host reference is secondary and does not count as another isotope atom.
    ``placement`` is the source-derived name boundary at which the descriptor
    is inserted; the caller composes this with parent isotope/stereo text.
    """

    disposition: str
    reason: str
    suffix_kind: str
    parts: tuple[SuffixIsotopePart, ...] = ()
    hydrogen_host_references: tuple[tuple[int, int, str], ...] = ()
    owned_atom_ids: frozenset[int] = frozenset()
    placement: str = ""
    rules: tuple[str, ...] = RULES

    @property
    def valid(self) -> bool:
        if self.disposition != "supported":
            return False
        ids = tuple(part.atom_id for part in self.parts)
        return (bool(ids) and len(ids) == len(set(ids))
                and frozenset(ids) == self.owned_atom_ids
                and bool(self.placement))

@dataclass(frozen=True, slots=True)
class AcylSuffixIsotopeComponent:
    """Mapped writer input plus the separate suffix-role proof."""

    component: object
    evidence: SuffixIsotopeEvidence
    embedded_suffix_atoms: frozenset[int]
    original_heavy_atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class AcylSuffixIsotopeRendering:
    """Name and atom-level proof from the shared mapped component writer."""

    text: str
    component: object
    role_evidence: SuffixIsotopeEvidence
    primary_isotope_atom_ids: frozenset[int]
    isotope_hydrogen_bindings: tuple[tuple[int, int, str], ...]
    decoration: object | None = None
    isotope_coverage: object | None = None


@dataclass(frozen=True, slots=True)
class AcylSuffixIsotopeDecision(AnnotatedAcylReplacementDecision):
    """Complete fixed-suffix decision with separately typed suffix roles."""

    suffix_role_evidence: SuffixIsotopeEvidence = SuffixIsotopeEvidence(
        "outside-scope", "missing suffix-role evidence", "")


def _fixed_center(mol, proof, path):
    """Return the source-fixed suffix center after validating its parent locant."""
    centers = tuple(getattr(getattr(proof, "decision", None), "principal_centers", ()))
    if len(centers) != 1:
        return None
    center = centers[0]
    if center not in path or path.index(center) not in {0, len(path) - 1}:
        return None
    try:
        suffix_locant = int(proof.suffix_locant)
    except (TypeError, ValueError):
        return None
    if suffix_locant != path.index(center) + 1:
        return None
    if mol.GetAtomWithIdx(center).GetAtomicNum() != 6:
        return None
    return center


def _valid_explicit_hydrogen(mol, atom_id, *, isotope=False):
    atom = mol.GetAtomWithIdx(atom_id)
    return (atom.GetAtomicNum() == 1 and atom.GetFormalCharge() == 0
            and atom.GetNumRadicalElectrons() == 0 and atom.GetDegree() == 1
            and (not isotope or atom.GetIsotope() > 0)
            and all(bond.GetBondTypeAsDouble() == 1.0
                    for bond in atom.GetBonds()))


def _pending(kind: str, reason: str) -> SuffixIsotopeEvidence:
    return SuffixIsotopeEvidence("pending", reason, kind)


def _outside(kind: str, reason: str) -> SuffixIsotopeEvidence:
    return SuffixIsotopeEvidence("outside-scope", reason, kind)


def recognize_fixed_suffix_isotopes(mol, proof) -> SuffixIsotopeEvidence:
    """Recognize closed suffix isotope roles against a fixed P-51 proof.

    ``proof`` is an ``AcylReplacementProof`` from the isotope-cleared graph.
    The original graph must preserve its atom indices.  This function neither
    chooses a path nor renders the whole name.  Labels outside the parent and
    exact terminal suffix are rejected or returned as pending.
    """
    from .acyl_replacement_methods import AcylReplacementProof

    if not isinstance(proof, AcylReplacementProof) or mol is None:
        return _outside("", "missing fixed acyl-replacement proof")
    kind = proof.suffix_kind
    if kind not in {"acid", "amide", "nitrile"}:
        return _outside(kind, "suffix kind is outside the fixed acyl grammar")
    path = tuple(proof.parent_path)
    if (not path or len(set(path)) != len(path)
            or any(not isinstance(i, int) or isinstance(i, bool)
                   or not 0 <= i < mol.GetNumAtoms() for i in path)):
        return _outside(kind, "invalid original-atom parent path")
    roles = tuple(getattr(proof.decision, "component_roles", ()))
    suffix_roles = [role for role in roles if role and role[0] == "principal_suffix"]
    if len(suffix_roles) != 1:
        return _outside(kind, "principal suffix atom ownership is not unique")
    suffix_atoms = frozenset(suffix_roles[0][1])
    expected_elements = {"acid": (8, 8), "amide": (7, 8), "nitrile": (7,)}[kind]
    if (not suffix_atoms or len(suffix_atoms) != len(expected_elements)
            or tuple(sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in suffix_atoms))
            != tuple(sorted(expected_elements))):
        return _outside(kind, "suffix atom roles do not match the fixed proof")
    center = _fixed_center(mol, proof, path)
    if center is None:
        return _outside(kind, "fixed suffix center/parent locant is inconsistent")
    if any(mol.GetBondBetweenAtoms(center, atom_id) is None for atom_id in suffix_atoms):
        return _outside(kind, "suffix atoms do not attach to the source-fixed parent carbon")

    # Every isotope owner must be either on the numbered parent or in the
    # narrowly typed suffix unit. Hydrogen labels additionally require one
    # neutral, single-bonded heavy-atom host.
    parent_atoms = frozenset(path)
    for atom in mol.GetAtoms():
        if not atom.GetIsotope():
            continue
        atom_id = atom.GetIdx()
        if atom.GetAtomicNum() > 1:
            if atom_id not in parent_atoms | suffix_atoms:
                return _pending(kind, "isotope-bearing heavy atom lies outside fixed parent and suffix")
            continue
        if atom.GetAtomicNum() != 1 or not _valid_explicit_hydrogen(mol, atom_id, isotope=True):
            return _pending(kind, "isotope hydrogen lacks neutral single-bonded explicit ownership")
        hosts = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(hosts) != 1 or hosts[0] not in parent_atoms | suffix_atoms:
            return _pending(kind, "isotope hydrogen host lies outside fixed parent and suffix")

    # Verify the exact characteristic-group atom topology, including the
    # external heteroatom role.  The carbonyl carbon remains in the parent.
    if kind == "acid":
        from .oxygen_characteristic_suffixes import oxygen_group
        from types import SimpleNamespace

        typed = oxygen_group(mol, SimpleNamespace(kind="acid", center=center))
        if typed is None or frozenset(typed.oxygen_atoms) != suffix_atoms:
            return _outside(kind, "acid O/OH roles are not structurally typed")
        role_by_atom = {idx: role for role, idx in typed.roles}
        isotope_oxygen_ids = [idx for idx in suffix_atoms
                              if mol.GetAtomWithIdx(idx).GetIsotope()]
        isotope_hydrogens = [idx for idx, atom in enumerate(mol.GetAtoms())
                             if atom.GetAtomicNum() == 1 and atom.GetIsotope()
                             and any(n.GetIdx() in suffix_atoms for n in atom.GetNeighbors())]
        if isotope_oxygen_ids:
            # P-82's acid oxygen examples require an independently typed
            # hydroxyl-H role to distinguish the two carboxyl oxygens. The
            # exact systematic suffix insertion/count grammar is not claimed
            # by this helper, so keep it pending for the next closed rule.
            return _pending(kind, "acid suffix O isotope needs a closed suffix-local role/count form")
        if not isotope_hydrogens:
            return _outside(kind, "no isotope-modified acid suffix atom")
        if len(isotope_hydrogens) != 1:
            return _pending(kind, "multiple isotope hydrogens on acid suffix are unclosed")
        h_id = isotope_hydrogens[0]
        if not _valid_explicit_hydrogen(mol, h_id, isotope=True):
            return _pending(kind, "acid isotope hydrogen is not a neutral single-bonded explicit H")
        host_ids = [n.GetIdx() for n in mol.GetAtomWithIdx(h_id).GetNeighbors()
                    if n.GetAtomicNum() > 1]
        if len(host_ids) != 1 or role_by_atom.get(host_ids[0]) != "acid_hydroxyl_oxygen":
            return _outside(kind, "acid isotope H is not bonded to the typed hydroxyl oxygen")
        if (mol.GetAtomWithIdx(h_id).GetIsotope() <= 0
                or mol.GetAtomWithIdx(host_ids[0]).GetIsotope()):
            return _pending(kind, "isotope-modified acid O-H combination is outside this grammar")
        part = SuffixIsotopePart(
            "H", mol.GetAtomWithIdx(h_id).GetIsotope(), "O", h_id,
            "acid_hydroxyl_hydrogen", "front-of-acid-component",
        )
        evidence = SuffixIsotopeEvidence(
            "supported", "P-82 O-locant hydroxyl-H role on systematic acid suffix",
            kind, (part,), ((h_id, host_ids[0], "acid_hydroxyl_oxygen"),),
            frozenset({h_id}), "front-of-acid-component",
        )
        return evidence if evidence.valid else _outside(kind, "acid role record failed ownership validation")

    if kind == "amide":
        nitrogen_ids = [idx for idx in suffix_atoms
                        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7]
        if len(nitrogen_ids) != 1:
            return _outside(kind, "primary amide has no unique suffix nitrogen")
        nitrogen_id = nitrogen_ids[0]
        nitrogen = mol.GetAtomWithIdx(nitrogen_id)
        if (mol.GetBondBetweenAtoms(center, nitrogen_id).GetBondTypeAsDouble() != 1.0
                or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
                or sum(n.GetAtomicNum() > 1 for n in nitrogen.GetNeighbors()) != 1
                or nitrogen.GetTotalValence() != 3):
            return _outside(kind, "amide nitrogen is not a neutral primary terminal N")
        all_n_hydrogens = [n.GetIdx() for n in nitrogen.GetNeighbors()
                           if n.GetAtomicNum() == 1]
        if nitrogen.GetTotalNumHs(includeNeighbors=True) != 2:
            return _outside(kind, "amide N does not have exactly two hydrogen positions")
        if any(not _valid_explicit_hydrogen(mol, h_id)
               for h_id in all_n_hydrogens):
            return _pending(kind, "amide N has a non-neutral or non-single-bonded explicit H")
        parts = []
        refs = []
        for atom_id in suffix_atoms:
            atom = mol.GetAtomWithIdx(atom_id)
            if atom.GetIsotope():
                if atom_id != nitrogen_id:
                    return _pending(kind, "amide carbonyl-O isotope insertion is not closed")
                parts.append(SuffixIsotopePart(
                    "N", atom.GetIsotope(), "N", atom_id,
                    "primary_amide_nitrogen", "front-of-one-word-amide-name",
                ))
        isotope_hydrogens = [h_id for h_id in all_n_hydrogens
                             if mol.GetAtomWithIdx(h_id).GetIsotope()]
        if len(isotope_hydrogens) > 2:
            return _pending(kind, "isotope-H count exceeds the two primary amide N positions")
        for h_id in isotope_hydrogens:
            parts.append(SuffixIsotopePart(
                "H", mol.GetAtomWithIdx(h_id).GetIsotope(), "N", h_id,
                "amide_nitrogen_hydrogen", "front-of-one-word-amide-name",
            ))
            refs.append((h_id, nitrogen_id, "amide_nitrogen"))
        if not parts:
            return _outside(kind, "no isotope-modified primary-amide suffix atom")
        if any(atom.GetIsotope() and atom.GetAtomicNum() > 1
               and atom.GetIdx() not in suffix_atoms | parent_atoms
               for atom in mol.GetAtoms()):
            return _outside(kind, "unexpected heavy atom outside parent and suffix")
        if any(part.mass <= 0 for part in parts):
            return _outside(kind, "invalid amide isotope mass")
        evidence = SuffixIsotopeEvidence(
            "supported", "P-66 N locant plus P-82 one-word suffix descriptor",
            kind, tuple(parts), tuple(refs), frozenset(p.atom_id for p in parts),
            "front-of-one-word-amide-name",
        )
        return evidence if evidence.valid else _outside(kind, "amide role record failed ownership validation")

    # Nitrile N has a unique group atom but this bounded tranche has no
    # source-closed way to place its isotope descriptor inside carbonitrile.
    if any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in suffix_atoms):
        return _pending(kind, "nitrile-N suffix-local isotope insertion remains unproved")
    return _outside(kind, "no isotope-modified nitrile suffix atom")


def mapped_suffix_isotope_component(mol, proof) -> AcylSuffixIsotopeComponent | None:
    """Build a complete mapped component for the source-closed suffix forms.

    The component parent is the already-constructed, unmodified complete name
    from ``AcylReplacementProof``.  Parent atoms keep their Arabic locants;
    suffix N or O and its isotope-H use conventional letter locants.  Any
    unmodified external carbonyl oxygen is an embedded, separately owned
    suffix atom.  The returned component is suitable for the existing mapped
    component writer, but callers must still verify whole-molecule isotope
    coverage and combine it with the structural proof.
    """
    from .component_mapping import MappedComponent
    from .acyl_replacement_methods import AcylReplacementProof

    if not isinstance(proof, AcylReplacementProof) or mol is None:
        return None
    evidence = recognize_fixed_suffix_isotopes(mol, proof)
    if not evidence.valid:
        return None
    path = tuple(proof.parent_path)
    center = _fixed_center(mol, proof, path)
    if center is None:
        return None
    roles = tuple(getattr(proof.decision, "component_roles", ()))
    suffix_rows = [row for row in roles if row and row[0] == "principal_suffix"]
    if len(suffix_rows) != 1:
        return None
    suffix_atoms = frozenset(suffix_rows[0][1])
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    parent_atoms = frozenset(path)
    if parent_atoms & suffix_atoms or parent_atoms | suffix_atoms != heavy:
        return None

    mapped_atoms = set(parent_atoms)
    embedded = set()
    locants = {idx: str(position) for position, idx in enumerate(path, 1)}
    typed_component_evidence = None
    typed_component_mode = ""
    local_isotopes = frozenset()
    descriptor_parts = ()
    if proof.suffix_kind == "acid":
        from .oxygen_characteristic_suffixes import oxygen_group
        from types import SimpleNamespace

        typed = oxygen_group(mol, SimpleNamespace(kind="acid", center=center))
        if typed is None:
            return None
        roles_by_name = dict(typed.roles)
        hydroxyl = roles_by_name.get("acid_hydroxyl_oxygen")
        carbonyl = roles_by_name.get("carbonyl_oxygen")
        if hydroxyl is None or carbonyl is None or {hydroxyl, carbonyl} != set(suffix_atoms):
            return None
        if mol.GetAtomWithIdx(carbonyl).GetIsotope():
            return None
        mapped_atoms.add(hydroxyl)
        embedded.add(carbonyl)
        locants[hydroxyl] = "O"
        for hydrogen_id, host_id, role in evidence.hydrogen_host_references:
            if host_id != hydroxyl or role != "acid_hydroxyl_oxygen":
                return None
            mapped_atoms.add(hydrogen_id)
            locants[hydrogen_id] = "O"
        # P-82.2.4 gives the acid suffix-O hydrogen descriptor without a
        # subscript 1. Keep its original H owner and O host in a checked
        # component-parts record; an ordinary mapped-H token would add H1.
        from .acid_oxygen_isotope_methods import AcidOxygenIsotopeRecord
        hydrogen_id = evidence.parts[0].atom_id
        typed_component_evidence = AcidOxygenIsotopeRecord(
            (("H", evidence.parts[0].mass, "O", True),),
            (hydrogen_id,),
            tuple((h, host, "hydroxyl_oxygen")
                  for h, host, _role in evidence.hydrogen_host_references),
            frozenset({hydrogen_id}),
            status="source-rule",
        )
        local_isotopes = frozenset({hydrogen_id})
        descriptor_parts = typed_component_evidence.parts
        typed_component_mode = "component-parts"
    elif proof.suffix_kind == "amide":
        nitrogen = next((idx for idx in suffix_atoms
                         if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7), None)
        oxygen = next((idx for idx in suffix_atoms
                       if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8), None)
        if nitrogen is None or oxygen is None:
            return None
        if mol.GetAtomWithIdx(oxygen).GetIsotope():
            return None
        mapped_atoms.add(nitrogen)
        locants[nitrogen] = "N"
        embedded.add(oxygen)
        for hydrogen_id, host_id, role in evidence.hydrogen_host_references:
            if host_id != nitrogen or role != "amide_nitrogen":
                return None
            mapped_atoms.add(hydrogen_id)
            locants[hydrogen_id] = "N"
    else:
        return None

    # Explicit isotope-H atoms inherit the proved locant of their mapped
    # parent or suffix host. Host tuples remain secondary references only.
    from .component_mapping import hydrogen_bindings
    complete_locants = hydrogen_bindings(mol, locants)
    for atom_id, locant in complete_locants.items():
        if atom_id not in locants:
            atom = mol.GetAtomWithIdx(atom_id)
            if atom.GetAtomicNum() == 1 and atom.GetIsotope():
                mapped_atoms.add(atom_id)
                locants[atom_id] = locant

    mapped_heavy = frozenset(
        idx for idx in mapped_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
    )
    if any(atom.GetIsotope() and atom.GetIdx() not in mapped_atoms
           for atom in mol.GetAtoms()):
        return None
    if (mapped_heavy & frozenset(embedded)
            or mapped_heavy | frozenset(embedded) != heavy
            or set(locants) != mapped_atoms
            or any(a.GetIsotope() and a.GetIdx() not in mapped_atoms
                   for a in mol.GetAtoms())):
        return None
    component = MappedComponent(
        parent=proof.decision.name,
        atom_locants=tuple((idx, locants[idx]) for idx in sorted(mapped_atoms)),
        atoms=frozenset(mapped_atoms),
        embedded_atoms=frozenset(embedded),
        systematic_parent=True,
        local_isotope_atoms=local_isotopes,
        isotope_descriptor_parts=descriptor_parts,
        isotope_role_evidence=typed_component_evidence,
        isotope_role_evidence_mode=typed_component_mode,
    )
    return AcylSuffixIsotopeComponent(
        component, evidence, frozenset(embedded), heavy,
    )


def render_mapped_suffix_isotope_component(mol, Chem, proof) -> AcylSuffixIsotopeRendering | None:
    """Render the fixed suffix component and prove every original label owner."""
    from .component_mapping import (
        _mapped_isotope_descriptor_plan,
        _valid_isotope_role_evidence,
        decorate_component,
    )

    result = mapped_suffix_isotope_component(mol, proof)
    if result is None or not result.evidence.valid:
        return None
    component = result.component
    locants = dict(component.atom_locants)
    if (set(locants) != set(component.atoms)
            or not _valid_isotope_role_evidence(mol, component)):
        return None
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
    # The shared mapped writer owns isotope-token grouping. Its typed
    # hydrogen-host operation coalesces repeated locants at one characteristic
    # N while retaining every original H atom claim and numeric parent locant.
    text = decorate_component(mol, Chem, component)
    if plan is None or not text:
        return None
    _descriptor_parts, ordinary_claims = plan
    primary = {row[0] for row in ordinary_claims if row[0] is not None}
    typed = set(result.evidence.owned_atom_ids)
    if primary | typed != {
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope()
    }:
        return None
    refs = result.evidence.hydrogen_host_references
    if any(mol.GetBondBetweenAtoms(h, host) is None for h, host, _role in refs):
        return None
    return AcylSuffixIsotopeRendering(
        text, component, result.evidence, frozenset(primary | typed), refs,
    )


def decorate_fixed_suffix_isotope_proof(mol, Chem, proof) -> AcylSuffixIsotopeDecision | None:
    """Return a complete annotated decision for a closed fixed suffix form.

    The suffix role record is structural. Descriptor text and primary isotope
    claims come from the shared mapped component writer; only the acid
    hydroxyl-H token, whose P-82.2.4 form omits H1, is claimed from its local
    typed component-parts role. Parent-bound isotope H is mapped to its host
    and remains owned by the ordinary writer.
    """
    from .acyl_replacement_methods import AcylReplacementProof
    from .component_annotation_evidence import (
        MappedComponentDecoration, MappedStereoAtomEvidence,
        MappedStereoBondEvidence,
    )
    from .component_mapping import (
        _mapped_isotope_descriptor_plan, _mapped_stereochemistry_plan,
        has_annotations, hydrogen_bindings,
    )
    from .isotope_component_coverage import (
        component_record_from_atom_map, validate_isotope_component_coverage,
    )

    if not isinstance(proof, AcylReplacementProof) or mol is None:
        return None
    rendered = render_mapped_suffix_isotope_component(mol, Chem, proof)
    mapped = mapped_suffix_isotope_component(mol, proof)
    if rendered is None or mapped is None or has_annotations(mol, mapped.embedded_suffix_atoms):
        return None
    decision = proof.decision
    path = tuple(proof.parent_path)
    center = _fixed_center(mol, proof, path)
    component = mapped.component
    locants = dict(component.atom_locants)
    if (center is None or set(locants) != set(component.atoms)
            or decision.method != "skeletal-replacement"
            or tuple(dict(decision.atom_locants)) != path):
        return None

    isotope_plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
    stereo_plan = _mapped_stereochemistry_plan(mol, Chem, component, locants)
    if isotope_plan is None or stereo_plan is None:
        return None
    _parts, ordinary_claims = isotope_plan
    ordinary_ids = tuple(row[0] for row in ordinary_claims)
    if len(ordinary_ids) != len(set(ordinary_ids)):
        return None
    evidence = mapped.evidence
    # Amide N/H facts attest to the ordinary writer's claims. Acid hydroxyl
    # H is a locally emitted token and is added once here because its ordinary
    # writer plan deliberately excludes local isotope atoms.
    local_ids = (evidence.owned_atom_ids if proof.suffix_kind == "acid"
                 else frozenset())
    if set(ordinary_ids) & set(local_ids):
        return None
    descriptor_ids = tuple(ordinary_ids) + tuple(sorted(local_ids))
    if len(descriptor_ids) != len(set(descriptor_ids)):
        return None
    expected_ids = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope()}
    if set(descriptor_ids) != expected_ids:
        return None
    printed_locants = {
        row[0]: row[4] for row in ordinary_claims
    }
    for part in evidence.parts:
        if part.atom_id in local_ids:
            printed_locants[part.atom_id] = part.locant or None
    record = component_record_from_atom_map(
        mol,
        component_id="acyl-replacement:fixed-suffix-isotope",
        component_text=rendered.text,
        atom_ids=component.atoms,
        atom_locants=tuple((idx, locants[idx] or None)
                           for idx in sorted(component.atoms)),
        descriptor_atom_ids=descriptor_ids,
        printed_locants=printed_locants,
    )
    coverage = validate_isotope_component_coverage(mol, (record,))
    if not coverage.complete:
        return None

    _stereo_text, stereo_atom_rows, stereo_bond_rows = stereo_plan
    decoration = MappedComponentDecoration(
        text=rendered.text,
        isotope_record=record,
        stereo_atoms=tuple(MappedStereoAtomEvidence(*row)
                           for row in stereo_atom_rows),
        stereo_bonds=tuple(MappedStereoBondEvidence((a, b), locant, descriptor)
                           for a, b, locant, descriptor in stereo_bond_rows),
        evidence_scope=component.atoms,
    )
    equivalent_maps = []
    for equivalent_path in proof.equivalent_paths or (path,):
        equivalent_path = tuple(equivalent_path)
        if set(equivalent_path) != set(path) or len(equivalent_path) != len(path):
            return None
        equivalent_maps.append(tuple((idx, str(position))
                                     for position, idx in enumerate(equivalent_path, 1)))
    equivalent_maps = tuple(dict.fromkeys(equivalent_maps))
    # Every explicit isotope-H host is a mapped original atom. hydrogen_bindings
    # supplies the corresponding component locant without creating a second
    # descriptor owner for the host.
    all_locants = hydrogen_bindings(mol, locants)
    h_bindings = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or not atom.GetIsotope():
            continue
        hosts = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if len(hosts) != 1 or hosts[0] not in locants:
            return None
        h_id, host = atom.GetIdx(), hosts[0]
        if h_id not in component.atoms or h_id not in all_locants:
            return None
        h_bindings.append((h_id, host, all_locants[h_id]))

    suffix_rule = "/P-82.2.1/P-82.2.4/P-82.2.5/P-82.6.1/P-82.6.4"
    if evidence.suffix_kind == "amide":
        symbols = {part.symbol for part in evidence.parts}
        h_masses = {part.mass for part in evidence.parts if part.symbol == "H"}
        if len(symbols) > 1:
            suffix_rule += "/P-82.3.1"
        if len(h_masses) > 1:
            suffix_rule += "/P-82.3.2"
    return AcylSuffixIsotopeDecision(
        name=rendered.text,
        method=decision.method,
        principal_centers=decision.principal_centers,
        atoms=decision.atoms,
        rule=decision.rule + suffix_rule,
        atom_locants=decision.atom_locants,
        heterounits=decision.heterounits,
        component_roles=decision.component_roles,
        functional_parent_frontier=decision.functional_parent_frontier,
        structural_proof=proof,
        decoration=decoration,
        isotope_coverage=coverage,
        equivalent_atom_locants=equivalent_maps,
        fixed_suffix_locant=str(proof.suffix_locant),
        original_heavy_ownership=frozenset(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
        ),
        isotope_hydrogen_bindings=tuple(sorted(h_bindings)),
        suffix_role_evidence=evidence,
    )

