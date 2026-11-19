"""Typed isotope roles for source-backed carboxylic-acid components.

P-82 gives retained acetic-acid examples with isotope-modified carboxyl
oxygens and an isotopic hydrogen on the hydroxyl oxygen. This module keeps
the isotope atoms that contribute descriptor tokens separate from the
secondary oxygen reference that locates an isotopic hydrogen.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds

RULES = ("P-82.2.1", "P-82.2.4", "P-82.3.1", "P-82.3.2", "P-82.6.4",
         "P-65.1.1.1.2", "P-65.1.2.1")


@dataclass(frozen=True, slots=True)
class AcidOxygenIsotopeRecord:
    """One acid component's local descriptor plus exact original-atom roles.

    ``parts`` uses ``(symbol, mass, locant, omit_single_count)`` entries.
    ``part_atom_ids`` is parallel to ``parts``: every entry owns exactly one
    isotope-modified atom. ``hydrogen_host_references`` records the oxygen
    used as the locant for each oxygen-bound isotopic hydrogen; the oxygen is
    not counted again by that reference.
    """

    parts: tuple[tuple[str, int, str, bool], ...]
    part_atom_ids: tuple[int, ...]
    hydrogen_host_references: tuple[tuple[int, int, str], ...]
    owned_atom_ids: frozenset[int]
    rule_evidence: tuple[str, ...] = RULES
    status: str = "source-rule"

    @property
    def valid(self):
        return bool(self.parts) and len(self.parts) == len(self.part_atom_ids) \
            and len(set(self.part_atom_ids)) == len(self.part_atom_ids) \
            and frozenset(self.part_atom_ids) == self.owned_atom_ids


def retained_acid_oxygen_record(mol, group):
    """Return typed retained-acid isotope roles, or ``None`` if unproved.

    The atom-role grammar is mass-generic. It admits one isotope-modified H
    on the acid hydroxyl oxygen and isotope-modified atoms on either
    carboxyl oxygen. Repeated oxygen nuclides are separate primary entries so
    the component formatter emits a count, while the H locant's oxygen
    reference remains secondary metadata.
    """
    if mol is None or getattr(group, "kind", None) != "acid":
        return None
    from .oxygen_characteristic_suffixes import oxygen_group

    typed = oxygen_group(mol, group)
    if typed is None or not _valid_acid_oxygen_roles(mol, group, typed):
        return None
    roles = dict(typed.roles)
    hydroxyl = roles.get("acid_hydroxyl_oxygen")
    carbonyl = roles.get("carbonyl_oxygen")
    if hydroxyl is None or carbonyl is None:
        return None
    h_atoms = tuple(sorted(
        n.GetIdx() for n in mol.GetAtomWithIdx(hydroxyl).GetNeighbors()
        if n.GetAtomicNum() == 1 and n.GetIsotope()
    ))
    if len(h_atoms) > 1:
        return None
    oxygen_masses = tuple(mol.GetAtomWithIdx(i).GetIsotope()
                          for i in (hydroxyl, carbonyl))
    if any(oxygen_masses) and not h_atoms:
        # The cited retained-acid oxygen examples pair O modification with an
        # isotope locant on the acid hydroxyl hydrogen. Without it, the
        # non-numbered carboxyl oxygen role is not identified by this rule.
        return None
    parts = []
    atom_ids = []
    references = []
    if h_atoms:
        h_idx = h_atoms[0]
        h_mass = mol.GetAtomWithIdx(h_idx).GetIsotope()
        if h_mass <= 0:
            return None
        o_mass = mol.GetAtomWithIdx(hydroxyl).GetIsotope()
        o_locant = f"{o_mass}O" if o_mass else "O"
        parts.append(("H", h_mass, o_locant, True))
        atom_ids.append(h_idx)
        references.append((h_idx, hydroxyl, "hydroxyl_oxygen"))
    for role, oxygen_idx in (("acid_hydroxyl_oxygen", hydroxyl),
                             ("carbonyl_oxygen", carbonyl)):
        mass = mol.GetAtomWithIdx(oxygen_idx).GetIsotope()
        if mass:
            if mass <= 0:
                return None
            parts.append(("O", mass, "", False))
            atom_ids.append(oxygen_idx)
    if not parts:
        return None
    # Descriptor symbols are ordered alphabetically, and nuclides of one
    # element by increasing mass (P-82.3). O-role order does not affect output.
    ordered = sorted(zip(parts, atom_ids), key=lambda pair: (pair[0][0], pair[0][1], pair[0][2]))
    record = AcidOxygenIsotopeRecord(
        tuple(part for part, _ in ordered),
        tuple(atom for _, atom in ordered),
        tuple(references),
        frozenset(atom_ids),
        status=("clause-derived-count" if any(
            part[0] == "O" and sum(1 for p in parts if p[:2] == part[:2]) > 1
            for part in parts
        ) else "source-rule"),
    )
    return record if record.valid else None


def retained_acid_component_text(base_name, record):
    """Render a retained acid component from typed entries, never by parsing."""
    if not base_name or record is None or not record.valid:
        return ""
    tokens = []
    i = 0
    while i < len(record.parts):
        symbol, mass, locant, omit_single_count = record.parts[i]
        if symbol == "H":
            token = f"{locant}-{mass}H"
            tokens.append(token)
            i += 1
            continue
        count = 1
        j = i + 1
        while j < len(record.parts) and record.parts[j][:2] == (symbol, mass):
            count += 1
            j += 1
        tokens.append(f"{mass}{symbol}" + (str(count) if count > 1 else ""))
        i = j
    return ("(" + ",".join(tokens) + ")" if tokens else "") + base_name


def systematic_acid_hydrogen_record(mol, group):
    """Return the direct systematic-acid hydroxyl-H suffix operation.

    P-82.2.4 exemplifies this placement on pentanoic acid. The rule's nuclide
    spelling is mass-generic, but oxygen-labeled systematic acid suffixes are
    outside this operation.
    """
    if mol is None or getattr(group, "kind", None) != "acid":
        return None
    from .oxygen_characteristic_suffixes import oxygen_group

    typed = oxygen_group(mol, group)
    if typed is None or not _valid_acid_oxygen_roles(mol, group, typed):
        return None
    roles = dict(typed.roles)
    hydroxyl = roles.get("acid_hydroxyl_oxygen")
    carbonyl = roles.get("carbonyl_oxygen")
    if hydroxyl is None or carbonyl is None:
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in (hydroxyl, carbonyl)):
        return None
    h_atoms = tuple(n.GetIdx() for n in mol.GetAtomWithIdx(hydroxyl).GetNeighbors()
                    if n.GetAtomicNum() == 1 and n.GetIsotope())
    if len(h_atoms) != 1:
        return None
    h_idx = h_atoms[0]
    mass = mol.GetAtomWithIdx(h_idx).GetIsotope()
    if mass <= 0:
        return None
    return AcidOxygenIsotopeRecord(
        (("H", mass, "", True),), (h_idx,), ((h_idx, hydroxyl, "hydroxyl_oxygen"),),
        frozenset({h_idx}), status="source-rule-generalized-nuclide",
    )


def _valid_acid_oxygen_roles(mol, group, typed):
    """Validate the original acid atom and bond roles before recording them."""
    if mol is None:
        return False
    if (any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetChiralTag().name != "CHI_UNSPECIFIED" or a.GetIsotope() < 0
            for a in iter_atoms(mol))
            or any(b.GetStereo().name != "STEREONONE" or b.GetBondDir().name != "NONE"
                   for b in iter_bonds(mol))):
        return False
    center = getattr(group, "center", -1)
    if center < 0 or center >= mol.GetNumAtoms():
        return False
    carbon = mol.GetAtomWithIdx(center)
    if (carbon.GetAtomicNum() != 6 or carbon.GetIsAromatic()
            or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons()
            or carbon.GetChiralTag().name != "CHI_UNSPECIFIED"):
        return False
    role_map = dict(typed.roles)
    hydroxyl = role_map.get("acid_hydroxyl_oxygen")
    carbonyl = role_map.get("carbonyl_oxygen")
    if hydroxyl is None or carbonyl is None or hydroxyl == carbonyl:
        return False
    neighbors = tuple(carbon.GetNeighbors())
    oxygens = tuple(n.GetIdx() for n in neighbors if n.GetAtomicNum() == 8)
    carbons = tuple(n.GetIdx() for n in neighbors if n.GetAtomicNum() == 6)
    if set(oxygens) != {hydroxyl, carbonyl} or len(carbons) > 1:
        return False
    if any(n.GetAtomicNum() not in {6, 8} for n in neighbors):
        return False
    if carbon.GetTotalValence() != 4:
        return False
    for oxygen_idx, expected_bond in ((hydroxyl, 1.0), (carbonyl, 2.0)):
        oxygen = mol.GetAtomWithIdx(oxygen_idx)
        bond = mol.GetBondBetweenAtoms(center, oxygen_idx)
        if (oxygen.GetAtomicNum() != 8 or oxygen.GetIsAromatic()
                or oxygen.GetFormalCharge() or oxygen.GetNumRadicalElectrons()
                or oxygen.GetTotalValence() != 2
                or bond is None or bond.GetBondTypeAsDouble() != expected_bond):
            return False
        if oxygen.GetChiralTag().name != "CHI_UNSPECIFIED":
            return False
        if bond.GetStereo().name != "STEREONONE" or bond.GetBondDir().name != "NONE":
            return False
        if oxygen.GetIsotope() < 0:
            return False
    for neighbor in mol.GetAtomWithIdx(carbonyl).GetNeighbors():
        if neighbor.GetIdx() != center:
            return False
    hydroxyl_atom = mol.GetAtomWithIdx(hydroxyl)
    h_neighbors = tuple(n for n in hydroxyl_atom.GetNeighbors() if n.GetAtomicNum() == 1)
    if len(h_neighbors) > 1:
        return False
    if any(n.GetIdx() != center and n.GetAtomicNum() != 1 for n in hydroxyl_atom.GetNeighbors()):
        return False
    for hydrogen in h_neighbors:
        bond = mol.GetBondBetweenAtoms(hydroxyl, hydrogen.GetIdx())
        if (hydrogen.GetFormalCharge() or hydrogen.GetNumRadicalElectrons()
                or hydrogen.GetIsotope() <= 0
                or hydrogen.GetDegree() != 1
                or bond is None or bond.GetBondTypeAsDouble() != 1.0
                or bond.GetStereo().name != "STEREONONE"
                or bond.GetBondDir().name != "NONE"):
            return False
    if hydroxyl_atom.GetTotalNumHs(includeNeighbors=True) != 1:
        return False
    return True


@dataclass(frozen=True, slots=True)
class AcidOxygenIsotopeComponent:
    """A bounded acid-component result with complete isotope ownership."""

    component_text: str
    center: int
    record: AcidOxygenIsotopeRecord
    rule: str


def recognize_linear_monoacid(mol, Chem):
    """Return the one typed acid group for a neutral saturated carbon chain.

    Only one unbranched acyclic carbon parent with one terminal carboxy group
    is admitted. Additional heteroatoms, functions, explicit hydrogens,
    stereochemical annotations, charges, and radicals decline. Carbon isotope
    labels are allowed for the parent mapper to own independently.
    """
    if mol is None or mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        return None
    if any(a.GetChiralTag().name != "CHI_UNSPECIFIED" or a.GetIsotope() < 0
           for a in iter_atoms(mol)):
        return None
    if any(b.GetStereo().name != "STEREONONE" or b.GetBondDir().name != "NONE"
           for b in iter_bonds(mol)):
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() not in {1, 6, 8} for a in atoms):
        return None
    carbons = tuple(a.GetIdx() for a in atoms if a.GetAtomicNum() == 6)
    if not carbons:
        return None
    for atom in atoms:
        if atom.GetAtomicNum() == 1:
            if not atom.GetIsotope():
                return None
            continue
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return None
    carbon_adjacency = {idx: [] for idx in carbons}
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetAtomicNum() == 6 and b.GetAtomicNum() == 6:
            if bond.GetBondTypeAsDouble() != 1:
                return None
            carbon_adjacency[a.GetIdx()].append(b.GetIdx())
            carbon_adjacency[b.GetIdx()].append(a.GetIdx())
    if any(len(neighbors) > 2 for neighbors in carbon_adjacency.values()):
        return None
    seen = set()
    stack = [carbons[0]]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(carbon_adjacency[idx])
    if seen != set(carbons) or sum(map(len, carbon_adjacency.values())) != 2 * (len(carbons) - 1):
        return None
    groups = []
    for carbon in carbons:
        oxygens = [n for n in mol.GetAtomWithIdx(carbon).GetNeighbors()
                   if n.GetAtomicNum() == 8]
        double = [o for o in oxygens
                  if mol.GetBondBetweenAtoms(carbon, o.GetIdx()).GetBondTypeAsDouble() == 2]
        singles = [o for o in oxygens
                   if mol.GetBondBetweenAtoms(carbon, o.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(double) == 1 and len(singles) == 1:
            hydroxy = [o for o in singles if o.GetTotalNumHs(includeNeighbors=True) == 1]
            if len(hydroxy) == 1:
                from types import SimpleNamespace
                groups.append(SimpleNamespace(kind="acid", center=carbon))
    if len(groups) != 1:
        return None
    center = groups[0].center
    if len(carbon_adjacency[center]) > 1:
        return None
    role_atoms = {center}
    acid_oxygen = [n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()
                   if n.GetAtomicNum() == 8]
    role_atoms.update(acid_oxygen)
    if any(a.GetAtomicNum() == 8 and a.GetIdx() not in role_atoms for a in atoms):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIdx() not in {
            n.GetIdx() for o_idx in acid_oxygen
            for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()
            if n.GetAtomicNum() == 1 and n.GetIsotope()
    } for a in atoms):
        return None
    return groups[0]


def acid_oxygen_isotope_component(mol, Chem, base_component):
    """Build a typed isotope component for an admitted linear monoacid.

    ``base_component`` is the selected, completed acid component supplied by
    the parent writer. It is never parsed or edited by string matching.
    """
    group = recognize_linear_monoacid(mol, Chem)
    if group is None:
        return None
    from .oxygen_characteristic_suffixes import retained_acid_base, systematic_acid_suffix

    if base_component.endswith(" acid") and base_component.rsplit(" ", 1)[0] in {"acetic", "formic"}:
        component_text, owned = retained_acid_base(mol, base_component, group)
        if not component_text:
            return None
        record = retained_acid_oxygen_record(mol, group)
        if record is None or record.owned_atom_ids != owned:
            return None
        return AcidOxygenIsotopeComponent(component_text, group.center, record,
                                          "P-82.2.4/P-82.6.4")
    if base_component.endswith("oic acid"):
        component_text, owned = systematic_acid_suffix(mol, base_component, group)
        if not component_text:
            return None
        record = systematic_acid_hydrogen_record(mol, group)
        if record is None or record.owned_atom_ids != owned:
            return None
        return AcidOxygenIsotopeComponent(component_text, group.center, record,
                                          "P-82.2.4/P-65.1.2.1")
    return None
