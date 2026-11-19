"""Graph-checked evidence for Blue Book P-44.4.1.3 and P-45.3.

Bonding number is a graph property, but citing a lambda descriptor is a
writer decision. These records keep those two claims separate: callers must
provide the writer's explicit lambda citation and, for P-45.3, exhaustive
prefix ownership. This module never turns high valence alone into a citation.
"""

from dataclasses import dataclass
import re
from collections import Counter

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get


@dataclass(frozen=True, slots=True)
class LambdaCitation:
    """A writer-cited lambda descriptor on one original-graph atom."""

    atom: int
    bonding_number: int
    citation_token: str


@dataclass(frozen=True, slots=True)
class ParentNonstandardProof:
    """Explicit writer citations for all nonstandard skeletal atoms."""

    citations: tuple[LambdaCitation, ...]


@dataclass(frozen=True, slots=True)
class PrefixLambdaCitation:
    """One cited lambda prefix group attached to a mapped parent.

    ``atoms`` owns the complete heavy-atom branch; ``host_atom`` establishes
    its direct attachment. ``role`` must be the caller's explicit prefix-role
    declaration. Cited atoms can be deeper in a group such as
    phosphanylmethyl, but only cited atoms that are themselves attached to
    ``host_atom`` contribute to P-45.3.
    """

    host_atom: int
    atoms: frozenset[int]
    citations: tuple[LambdaCitation, ...]
    role: str = "prefix"


@dataclass(frozen=True, slots=True)
class CharacteristicPrefixRole:
    """A source-recognized non-lambda prefix role tied to mapped atoms.

    ``emitted_prefix`` records the actual writer output at the attachment
    locant; it is checked against a source-specific writer, never used to
    discover ``kind``.  The role vocabulary is deliberately closed below.
    """

    kind: str
    inventory_kind: str
    center: int
    atoms: frozenset[int]
    host_atom: int
    emitted_prefix: str
    writer: str


@dataclass(frozen=True, slots=True)
class OwnedComponent:
    """An exhaustive non-parent heavy-atom component and its declared role."""

    atoms: frozenset[int]
    role: str
    role_evidence: str
    # Exact selected-group cores that the mapped writer included in a suffix.
    # These are required when the characteristic-group center itself lies in
    # an external component rather than at its parent attachment host.
    suffix_group_cores: tuple["SelectedSuffixCore", ...] = ()


@dataclass(frozen=True, slots=True)
class SelectedSuffixCore:
    """A recognized selected characteristic-group core mapped in a suffix."""

    kind: str
    center: int
    atoms: frozenset[int]
    embedded_attachments: tuple[tuple[int, frozenset[int]], ...] = ()


@dataclass(frozen=True, slots=True)
class SubstituentNonstandardProof:
    """Exhaustive ownership proof for nonstandard-bonding prefix groups."""

    groups: tuple[PrefixLambdaCitation, ...]
    components: tuple[OwnedComponent, ...]
    # Optional source construction map. Existing proof producers remain
    # compatible; the mapped-component builder fills both fields together.
    construction_atoms: frozenset[int] = frozenset()
    construction_embedded_atoms: frozenset[int] = frozenset()
    characteristic_prefix_roles: tuple[CharacteristicPrefixRole, ...] = ()


_CHARACTERISTIC_PREFIX_KINDS = {
    "sulfonic-acid": ("sulfo", "characteristic-sulfur-prefix"),
    "sulfinic-acid": ("sulfino", "characteristic-sulfur-prefix"),
    "sulfonamide": ("sulfamoyl", "characteristic-sulfur-prefix"),
    "phosphonic-acid": ("phosphono", "phosphorus-prefix"),
}
_CHARACTERISTIC_PREFIX_NUMBERS = {
    "sulfo": 6,
    "sulfino": 4,
    "sulfamoyl": 6,
    "phosphono": 5,
    "hydrazinesulfonyl": 6,
    "hydrazinesulfinyl": 4,
}


def _characteristic_prefix_spec(mol, group):
    kind = getattr(group, 'inventory_kind', getattr(group, 'kind', ''))
    if kind == 'hydrazide':
        ids = (getattr(group, 'center', None), *getattr(group, 'atoms', ()))
        if any(not isinstance(i, int) or isinstance(i, bool)
               or not 0 <= i < mol.GetNumAtoms() for i in ids):
            return None
        if mol.GetAtomWithIdx(group.center).GetAtomicNum() != 16:
            return None
        oxygens = sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in group.atoms)
        if oxygens in {1, 2}:
            return ('hydrazinesulfonyl' if oxygens == 2 else 'hydrazinesulfinyl',
                    'characteristic-sulfur-prefix')
        return None
    return _CHARACTERISTIC_PREFIX_KINDS.get(kind)


def _plain_characteristic_core(mol, atoms):
    """Limit the initial role scope to neutral, unannotated source cores."""
    for idx in atoms:
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < mol.GetNumAtoms():
            return False
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())):
            return False
        if any(neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope()
               for neighbor in atom.GetNeighbors()):
            return False
        if any(int(bond.GetStereo()) != 0 for bond in atom.GetBonds()):
            return False
    return True


def _recognized_characteristic_prefix_writer(mol, role):
    """Re-run the source-specific group writer for one typed prefix record."""
    from rdkit import Chem

    if role.writer == "characteristic-sulfur-prefix":
        from .sulfur_prefixes import characteristic_sulfur_prefix

        result = characteristic_sulfur_prefix(
            mol, Chem, role.center, role.host_atom, role.atoms,
        )
        return result if result is not None else None
    if role.writer == "phosphorus-prefix":
        from .branch.functional_groups import _phosphorus_prefix

        return _phosphorus_prefix(mol, role.center, role.host_atom)
    return None


def _characteristic_role_matches_inventory(mol, role):
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if not inventory.supported:
        return False
    matches = [group for group in inventory.groups
               if group.kind == role.inventory_kind
               and group.center == role.center
               and group.owner == role.host_atom
               and frozenset(group.atoms) == frozenset(role.atoms)]
    return len(matches) == 1


def characteristic_prefix_role(mol, host_atom, emitted_prefix, atoms,
                                inventory_groups):
    """Create a typed non-lambda role at a source-specific writer decision.

    This factory requires exactly one source inventory group and verifies the
    source-specific writer output. It does not inspect prefix text to select a
    group kind. Unsupported, decorated, or extended components return None.
    """
    branch = frozenset(atoms)
    if not branch or not isinstance(emitted_prefix, str) or not emitted_prefix:
        return None
    candidates = []
    for group in inventory_groups:
        spec = _characteristic_prefix_spec(mol, group)
        if spec is None or getattr(group, "owner", -1) != host_atom:
            continue
        core = frozenset(group.atoms)
        if (not core or core != branch or group.center not in core
                or not 0 <= group.center < mol.GetNumAtoms()
                or mol.GetBondBetweenAtoms(group.center, host_atom) is None
                or not _plain_characteristic_core(mol, core)):
            continue
        kind, writer = spec
        role = CharacteristicPrefixRole(
            kind, group.kind, int(group.center), core, int(host_atom),
            emitted_prefix, writer,
        )
        result = _recognized_characteristic_prefix_writer(mol, role)
        if writer == "characteristic-sulfur-prefix":
            if (not isinstance(result, tuple) or len(result) != 2
                    or result[0] != emitted_prefix
                    or frozenset(result[1]) != branch):
                continue
        elif result != emitted_prefix:
            continue
        candidates.append(role)
    return candidates[0] if len(candidates) == 1 else None


def _neutral_bonding_number(atom):
    if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
        return None
    if atom.GetIsAromatic():
        # P-14.1.2 gives sulfur its standard bonding number of two. RDKit
        # represents both bonds of neutral aromatic sulfur in thiophene-like
        # rings as 1.5, although the ring atom has two single bonds in its
        # Kekule structure. Admit only the ordinary divalent, two-ring-bond
        # case; charged or higher-valence aromatic sulfur still delegates.
        if (atom.GetAtomicNum() == 16 and atom.GetTotalValence() == 2
                and atom.GetDegree() == 2
                and all(bond.GetIsAromatic() and bond.IsInRing()
                        for bond in atom.GetBonds())):
            return 2
        return None
    if atom.GetSymbol() not in {"P", "S"}:
        return None
    valence = sum(
        bond.GetBondTypeAsDouble()
        for bond in atom.GetBonds()
        if bond.GetOtherAtom(atom).GetAtomicNum() > 1
    ) + atom.GetTotalNumHs(includeNeighbors=True)
    return int(valence) if valence.is_integer() else None


def _standard_bonding_number(atom):
    return {"P": 3, "S": 2}.get(atom.GetSymbol())


def _citation_matches(atom, number, token):
    if _neutral_bonding_number(atom) != number:
        return False
    if number not in {4, 5, 6} or number == _standard_bonding_number(atom):
        return False
    # The token is a narrow evidence field, not a complete generated name.
    match = re.fullmatch(r"(?:λ|lambda)([456])", str(token), flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) == number)


def parent_nonstandard_bonding_key(mol, atoms, locants, proof=None):
    """Return P-44 evidence, or None outside the closed sulfur cases.

    Standard or absent skeletal sulfur proves an empty feature. A higher
    bonding-number sulfur remains unknown unless the renderer supplies a
    matching lambda citation on that original atom.
    """
    parent = frozenset(atoms)
    sulfur = [mol.GetAtomWithIdx(i) for i in parent
              if mol.GetAtomWithIdx(i).GetAtomicNum() == 16]
    if proof is None:
        if not sulfur or all(_neutral_bonding_number(a) == 2 for a in sulfur):
            return ()
        return None
    if not isinstance(proof, ParentNonstandardProof):
        return None
    if any(_neutral_bonding_number(a) not in {2, 4, 5, 6}
           for a in sulfur):
        return None
    expected = {a.GetIdx() for a in sulfur
                if _neutral_bonding_number(a) not in {None, 2}}
    supplied = [c.atom for c in proof.citations]
    if len(supplied) != len(set(supplied)) or set(supplied) != expected:
        return None
    rows = []
    for citation in proof.citations:
        if citation.atom not in parent or citation.atom not in locants:
            return None
        atom = mol.GetAtomWithIdx(citation.atom)
        if atom.GetAtomicNum() != 16 or not _citation_matches(
                atom, citation.bonding_number, citation.citation_token):
            return None
        from .parent_selection import locant_key
        locant = locant_key(locants[citation.atom])
        if locant is None:
            return None
        rows.append((locant, citation.bonding_number))
    if not rows:
        return ()
    rows.sort()
    return (0, -len(rows), tuple(loc for loc, _ in rows),
            tuple(-number for _, number in rows))


def _native_component_graph(mol):
    """Read fresh native membership and adjacency for one proof validation.

    These primitive values stay local to that operation. Valence, isotopes,
    hydrogen counts, cited wording and ownership are validated independently.
    """
    if type(mol).__module__ != "rdkit.Chem.rdchem" or type(mol).__name__ not in {"Mol", "RWMol"}:
        atoms = tuple(iter_atoms(mol))
        return (tuple(atom.GetAtomicNum() for atom in atoms),
                tuple(tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
                      for atom in atoms))
    # RDKit orders an atom's neighbors by incident bond insertion order.
    # Walking that native bond sequence once reproduces those rows without
    # constructing a separate Python Atom wrapper for every adjacent atom.
    neighbors = [[] for _ in range(mol.GetNumAtoms())]
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbors[left].append(right)
        neighbors[right].append(left)
    return (tuple(atom.GetAtomicNum() for atom in iter_atoms(mol)),
            tuple(tuple(row) for row in neighbors))


def _existing_component_graph(mol):
    view = mol_cache_get(mol, "graph_view")
    if view is None:
        return None
    return view.atomic_nums, view.neighbors


def _heavy_component(mol, remaining, *, _graph=None):
    seed = next(iter(remaining))
    component, stack = set(), [seed]
    while stack:
        idx = stack.pop()
        if idx not in remaining:
            continue
        remaining.remove(idx)
        component.add(idx)
        if _graph is None:
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if n.GetAtomicNum() > 1 and n.GetIdx() in remaining)
        else:
            numbers, neighbors = _graph
            stack.extend(other for other in neighbors[idx]
                         if numbers[other] > 1 and other in remaining)
    return frozenset(component)


def _component_attachments(mol, atoms, parent, *, _graph=None):
    if _graph is not None:
        return tuple((idx, other) for idx in atoms for other in _graph[1][idx]
                     if other in parent)
    attachments = []
    for idx in atoms:
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetIdx() in parent:
                attachments.append((idx, neighbor.GetIdx()))
    return tuple(attachments)


def _external_components(mol, parent, *, _graph=None):
    if _graph is None:
        remaining = {a.GetIdx() for a in iter_atoms(mol)
                     if a.GetAtomicNum() > 1 and a.GetIdx() not in parent}
    else:
        remaining = {i for i, number in enumerate(_graph[0]) if number > 1 and i not in parent}
    components = []
    while remaining:
        components.append(_heavy_component(mol, remaining, _graph=_graph))
    return tuple(components)


def _external_component_data(mol, parent, *, _graph=None):
    """Reuse physical cuts on the API-owned graph across writer numberings.

    Only atom membership and boundary edges are shared. Ownership, locants,
    bonding numbers and citations remain live writer checks. Proof validation
    builds its own fresh native graph independently of this preparation.
    """
    cache = mol_cache_dict(mol, "nonstandard_external_components")
    data = cache.get(parent)
    if data is None:
        if _graph is None:
            _graph = _existing_component_graph(mol)
        components = _external_components(mol, parent, _graph=_graph)
        data = components, {
            atoms: _component_attachments(mol, atoms, parent, _graph=_graph)
            for atoms in components
        }
        cache[parent] = data
    return data


def build_substituent_nonstandard_proof(
        mol, parent_atoms, construction, prefixes, locants, selected_groups,
        prefix_lambda_groups=()):
    """Build exhaustive P-45.3 ownership from a completed mapped writer result.

    The helper deliberately does not infer lambda citations from the graph or
    the emitted text. ``prefix_lambda_groups`` must carry those explicit
    writer citations. A component that cannot be tied to a selected mapped
    suffix core or to an emitted locanted prefix makes the proof unavailable.
    """
    from .component_mapping import MappedComponent, fully_covered
    from .functional_inventory import CharacteristicGroup

    if not isinstance(construction, MappedComponent):
        return None
    parent = frozenset(parent_atoms)
    mapped = frozenset(construction.atoms)
    embedded = frozenset(construction.embedded_atoms)
    if (not parent or not parent <= mapped or mapped & embedded
            or dict(construction.atom_locants).keys() != mapped
            or not fully_covered(mol, mapped, embedded)):
        return None
    locants = dict(locants)
    if not parent <= locants.keys():
        return None
    emitted_prefix_counts = Counter(str(locant) for locant, _ in prefixes)
    if tuple(prefixes) != tuple(construction.prefixes):
        return None

    groups = tuple(selected_groups)
    if any(not isinstance(group, CharacteristicGroup) for group in groups):
        return None
    selected_groups_by_core = []
    for group in groups:
        core = frozenset(group.atoms)
        if (not core or group.center not in core or not core <= mapped
                or group.center not in mapped):
            return None
        selected_groups_by_core.append((group, core))

    proof_groups = tuple(prefix_lambda_groups)
    if any(not isinstance(group, PrefixLambdaCitation) for group in proof_groups):
        return None
    characteristic_roles = tuple(construction.characteristic_prefix_roles)
    if any(not isinstance(role, CharacteristicPrefixRole)
           for role in characteristic_roles):
        return None
    if len({(role.center, role.host_atom) for role in characteristic_roles}) != len(
            characteristic_roles):
        return None
    lambda_cited_atoms = {
        citation.atom for group in proof_groups for citation in group.citations
    }
    component_graph = _existing_component_graph(mol)
    external_components, attachments_by_component = _external_component_data(
        mol, parent, _graph=component_graph)
    role_components = {}
    for role in characteristic_roles:
        spec = _characteristic_prefix_spec(mol, role)
        if (spec != (role.kind, role.writer) or role.center not in role.atoms
                or role.center not in embedded or role.host_atom not in parent
                or not role.atoms <= (mapped | embedded) or not role.atoms <= embedded
                or not role.emitted_prefix):
            return None
        if (not _plain_characteristic_core(mol, role.atoms)
                or _neutral_bonding_number(mol.GetAtomWithIdx(role.center))
                != _CHARACTERISTIC_PREFIX_NUMBERS[role.kind]
                or set(role.atoms) & lambda_cited_atoms
                or any(set(group.atoms) & set(role.atoms) for group in proof_groups)):
            return None
        if not _characteristic_role_matches_inventory(mol, role):
            return None
        writer_result = _recognized_characteristic_prefix_writer(mol, role)
        if role.writer == "characteristic-sulfur-prefix":
            if (not isinstance(writer_result, tuple) or len(writer_result) != 2
                    or writer_result[0] != role.emitted_prefix
                    or frozenset(writer_result[1]) != role.atoms):
                return None
        elif writer_result != role.emitted_prefix:
            return None
        matching = [component for component in external_components
                    if component == role.atoms]
        if len(matching) != 1:
            return None
        attachments = attachments_by_component[role.atoms]
        if (len(attachments) != 1 or attachments[0] != (role.center, role.host_atom)
                or mol.GetBondBetweenAtoms(role.center, role.host_atom).GetBondTypeAsDouble() != 1):
            return None
        host_locant = str(locants.get(role.host_atom, ""))
        if not host_locant or (host_locant, role.emitted_prefix) not in tuple(prefixes):
            return None
        nonstandard = {
            idx for idx in role.atoms
            if mol.GetAtomWithIdx(idx).GetSymbol() in {"P", "S"}
            and _neutral_bonding_number(mol.GetAtomWithIdx(idx))
            != _standard_bonding_number(mol.GetAtomWithIdx(idx))
        }
        # Initial source-closed roles own the whole external component. This
        # prevents ordinary embedded groups, additional high-valence atoms,
        # and unknown remainder atoms from being masked by a characteristic
        # core claim.
        if nonstandard != {role.center}:
            return None
        role_components[role.atoms] = role
    components = []
    used_prefix_locs = Counter()
    for component_atoms in external_components:
        attachments = attachments_by_component[component_atoms]
        if len(attachments) != 1:
            return None
        _branch_atom, host = attachments[0]
        suffix_cores = []
        for group, core_atoms in selected_groups_by_core:
            if not (core_atoms <= mapped and
                    (group.center == host and bool(core_atoms & component_atoms)
                     or group.center in component_atoms)):
                continue
            core_nonstandard = set()
            for idx in component_atoms:
                if component_graph is not None and component_graph[0][idx] not in {15, 16}:
                    continue
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() not in {"P", "S"}:
                    continue
                number = _neutral_bonding_number(atom)
                if number not in {_standard_bonding_number(atom), 4, 5, 6}:
                    return None
                if number != _standard_bonding_number(atom):
                    core_nonstandard.add(idx)
            if not core_nonstandard <= core_atoms:
                # A selected suffix core cannot hide an unrelated
                # nonstandard-bonding atom elsewhere in the same component.
                return None
            embedded_attachments = []
            remaining_embedded = set(component_atoms & embedded)
            while remaining_embedded:
                branch = _heavy_component(mol, remaining_embedded, _graph=component_graph)
                attachments_to_core = _component_attachments(
                    mol, branch, core_atoms, _graph=component_graph)
                if (len(attachments_to_core) != 1
                        or mol.GetAtomWithIdx(attachments_to_core[0][1]).GetAtomicNum() != 7):
                    return None
                root = attachments_to_core[0][1]
                locant = str(locants.get(root, ""))
                if not locant or emitted_prefix_counts[locant] <= 0:
                    return None
                used_prefix_locs[locant] += 1
                embedded_attachments.append((root, branch))
            suffix_cores.append(SelectedSuffixCore(
                group.kind, int(group.center), core_atoms,
                tuple(embedded_attachments),
            ))
        suffix_cores = tuple(suffix_cores)
        if suffix_cores:
            components.append(OwnedComponent(
                component_atoms, "suffix", "suffix_ownership", suffix_cores,
            ))
            continue
        if component_atoms <= embedded:
            host_locant = str(locants.get(host, ""))
            if (not host_locant or emitted_prefix_counts[host_locant] <= 0):
                return None
            if component_atoms in role_components:
                role = role_components[component_atoms]
                if (host != role.host_atom or component_atoms != role.atoms
                        or (host_locant, role.emitted_prefix) not in tuple(prefixes)):
                    return None
            used_prefix_locs[host_locant] += 1
            components.append(OwnedComponent(
                component_atoms, "prefix", "prefix_mapping",
            ))
            continue
        # No generic "other" role may hide a nonstandard-bonding atom or
        # accidentally remove a component from the P-45.3 comparison.
        return None
    if used_prefix_locs != emitted_prefix_counts:
        return None
    if {frozenset(component.atoms) for component in components} != set(
            external_components):
        return None
    return SubstituentNonstandardProof(
        proof_groups, tuple(components), mapped, embedded, characteristic_roles,
    )


def substituent_nonstandard_bonding_key(mol, parent_atoms, locants, prefixes=None,
                                        proof=None, suffix_centers=()):
    """Return P-45.3 evidence only for explicit, exhaustive prefix ownership."""
    parent = frozenset(parent_atoms)
    if proof is None:
        outside_ps = [a for a in iter_atoms(mol)
                      if a.GetIdx() not in parent and a.GetSymbol() in {"P", "S"}]
        # No nonstandard-capable atom, or all such atoms have the standard
        # bonding number: no higher-bonding prefix can be present.
        if all(_neutral_bonding_number(a) == _standard_bonding_number(a)
               for a in outside_ps):
            return ()
        return None
    if not isinstance(proof, SubstituentNonstandardProof):
        return None
    if prefixes is None:
        return None
    mapped = frozenset(proof.construction_atoms)
    embedded = frozenset(proof.construction_embedded_atoms)
    if mapped:
        from .component_mapping import fully_covered

        if (mapped & embedded or not parent <= mapped
                or not fully_covered(mol, mapped, embedded)):
            return None
    prefix_locs = Counter(str(loc) for loc, _ in prefixes)
    declared = {}
    suffix_core_map = {}
    for component in proof.components:
        atoms = frozenset(component.atoms)
        evidence_for_role = {
            "prefix": "prefix_mapping",
            "suffix": "suffix_ownership",
            "other": "nonprefix_ownership",
        }
        if (component.role not in evidence_for_role
                or component.role_evidence != evidence_for_role[component.role]
                or not atoms
                or atoms & parent or atoms in declared):
            return None
        declared[atoms] = component.role
        suffix_core_map[atoms] = tuple(component.suffix_group_cores)
        if component.role == "prefix" and (mapped or embedded) and not atoms <= embedded:
            return None
    component_graph = _native_component_graph(mol)
    actual = set(_external_components(mol, parent, _graph=component_graph))
    if set(declared) != actual:
        return None
    characteristic_atoms = set()
    role_prefix_counts = Counter()
    all_cited_atoms = {
        citation.atom for group in proof.groups for citation in group.citations
    }
    for role in proof.characteristic_prefix_roles:
        if not isinstance(role, CharacteristicPrefixRole):
            return None
        if (_characteristic_prefix_spec(mol, role)
                != (role.kind, role.writer)
                or not role.atoms or role.center not in role.atoms
                or role.host_atom not in parent or role.host_atom not in locants
                or declared.get(frozenset(role.atoms)) != "prefix"
                or not frozenset(role.atoms) <= embedded):
            return None
        if (not _plain_characteristic_core(mol, role.atoms)
                or _neutral_bonding_number(mol.GetAtomWithIdx(role.center))
                != _CHARACTERISTIC_PREFIX_NUMBERS[role.kind]
                or set(role.atoms) & all_cited_atoms
                or any(set(group.atoms) & set(role.atoms) for group in proof.groups)):
            return None
        if not _characteristic_role_matches_inventory(mol, role):
            return None
        attachments = _component_attachments(mol, role.atoms, parent, _graph=component_graph)
        if (len(attachments) != 1 or attachments[0] != (role.center, role.host_atom)
                or mol.GetBondBetweenAtoms(role.center, role.host_atom).GetBondTypeAsDouble() != 1):
            return None
        writer_result = _recognized_characteristic_prefix_writer(mol, role)
        if role.writer == "characteristic-sulfur-prefix":
            if (not isinstance(writer_result, tuple) or len(writer_result) != 2
                    or writer_result[0] != role.emitted_prefix
                    or frozenset(writer_result[1]) != role.atoms):
                return None
        elif writer_result != role.emitted_prefix:
            return None
        if (str(locants[role.host_atom]), role.emitted_prefix) not in tuple(prefixes):
            return None
        high_atoms = {
            idx for idx in role.atoms
            if component_graph[0][idx] in {15, 16}
            and _neutral_bonding_number(mol.GetAtomWithIdx(idx))
            != _standard_bonding_number(mol.GetAtomWithIdx(idx))
        }
        if high_atoms != {role.center}:
            return None
        if characteristic_atoms & high_atoms:
            return None
        characteristic_atoms.update(high_atoms)
        role_prefix_counts[str(locants[role.host_atom])] += 1
    for atoms in actual:
        attachments = _component_attachments(mol, atoms, parent, _graph=component_graph)
        if len(attachments) != 1:
            return None
        role = declared[atoms]
        nonstandard_atoms = []
        for idx in atoms:
            if component_graph[0][idx] not in {15, 16}:
                continue
            atom = mol.GetAtomWithIdx(idx)
            number = _neutral_bonding_number(atom)
            standard = _standard_bonding_number(atom)
            if number not in {standard}:
                nonstandard_atoms.append(idx)
        if role == "suffix":
            cores = suffix_core_map[atoms]
            if any(core.center not in set(suffix_centers) for core in cores):
                return None
            owned_embedded = set()
            for core in cores:
                for root, branch in core.embedded_attachments:
                    branch = frozenset(branch)
                    if (not branch or branch & owned_embedded or branch & parent
                            or not branch <= embedded or root not in core.atoms
                            or mol.GetAtomWithIdx(root).GetAtomicNum() != 7
                            or not branch <= atoms):
                        return None
                    branch_links = _component_attachments(
                        mol, branch, core.atoms, _graph=component_graph)
                    if (len(branch_links) != 1 or branch_links[0][1] != root
                            or root not in locants):
                        return None
                    branch_locant = str(locants[root])
                    if prefix_locs[branch_locant] <= 0:
                        return None
                    prefix_locs[branch_locant] -= 1
                    owned_embedded.update(branch)
            if owned_embedded != set(atoms & embedded):
                return None
            if not (atoms - embedded) <= mapped:
                return None
            core_atoms = set().union(*(set(core.atoms) for core in cores))
            nonstandard_atoms = {
                idx for idx in atoms
                if component_graph[0][idx] in {15, 16}
                and _neutral_bonding_number(mol.GetAtomWithIdx(idx))
                != _standard_bonding_number(mol.GetAtomWithIdx(idx))
            }
            if (any(_neutral_bonding_number(mol.GetAtomWithIdx(idx))
                    not in {_standard_bonding_number(mol.GetAtomWithIdx(idx)), 4, 5, 6}
                    for idx in atoms if component_graph[0][idx] in {15, 16})
                    or not nonstandard_atoms <= core_atoms):
                return None
            if attachments[0][1] not in set(suffix_centers):
                # A sulfur/carbon suffix center may itself be outside the
                # parent path. Accept that role only with an exact selected
                # group core carried by the mapped suffix construction.
                valid_core = any(
                    core.center in atoms
                    and core.center not in parent
                    and core.center in set(suffix_centers)
                    and core.atoms
                    and core.center in core.atoms
                    and core.atoms <= mapped
                    and bool(core.atoms & atoms)
                    for core in cores
                )
                valid_core = valid_core or any(
                    core.center == attachments[0][1]
                    and core.center in set(suffix_centers)
                    and core.atoms <= mapped
                    and bool(core.atoms & atoms)
                    for core in cores
                )
                if not valid_core:
                    return None
            if any(
                    not core.kind or core.center not in core.atoms
                    or not core.atoms <= mapped
                    or not (core.center == attachments[0][1]
                            and core.center in parent and bool(core.atoms & atoms)
                            or core.center in atoms)
                    for core in cores):
                return None
        elif suffix_core_map[atoms]:
            return None
        if role == "other" and nonstandard_atoms:
            # Unknown high bonding cannot be hidden in an untyped "other"
            # component. A producer must establish prefix or suffix ownership.
            return None
    # The stored key is deliberately supplied only after ownership/citation
    # review; independently revalidate every cited branch and graph atom.
    rows = []
    used_components = set()
    for group in proof.groups:
        atoms = frozenset(group.atoms)
        if (group.role != "prefix" or not atoms or atoms & parent
                or atoms in used_components or group.host_atom not in parent
                or group.host_atom not in locants or not group.citations
                or declared.get(atoms) != "prefix"):
            return None
        used_components.add(atoms)
        remaining = set(atoms)
        component = _heavy_component(mol, remaining, _graph=component_graph)
        if remaining or component != atoms:
            return None
        attachments = _component_attachments(mol, atoms, parent, _graph=component_graph)
        if len(attachments) != 1 or attachments[0][1] != group.host_atom:
            return None
        cited_atoms = set()
        directly_attached_numbers = []
        for citation in group.citations:
            if citation.atom not in atoms or citation.atom in cited_atoms:
                return None
            cited_atoms.add(citation.atom)
            atom = mol.GetAtomWithIdx(citation.atom)
            if not _citation_matches(atom, citation.bonding_number,
                                     citation.citation_token):
                return None
            if any(neighbor.GetIdx() == group.host_atom
                   for neighbor in atom.GetNeighbors()):
                directly_attached_numbers.append(citation.bonding_number)
        from .parent_selection import locant_key
        locant = locant_key(locants[group.host_atom])
        if locant is None:
            return None
        cited_locant = str(locants[group.host_atom])
        if prefix_locs[cited_locant] <= 0:
            return None
        prefix_locs[cited_locant] -= 1
        if directly_attached_numbers:
            rows.append((max(directly_attached_numbers), locant))
    for locant, count in role_prefix_counts.items():
        if prefix_locs[locant] < count:
            return None
        prefix_locs[locant] -= count
    cited_atoms = {citation.atom for group in proof.groups
                   for citation in group.citations}
    for component_atoms, role in declared.items():
        if role != "prefix":
            continue
        for idx in component_atoms:
            if component_graph[0][idx] not in {15, 16}:
                continue
            atom = mol.GetAtomWithIdx(idx)
            number = _neutral_bonding_number(atom)
            standard = _standard_bonding_number(atom)
            if number is None:
                return None
            if number != standard and idx not in cited_atoms and idx not in characteristic_atoms:
                return None
    # Every higher-valence P/S atom in a prefix-owned branch must have an
    # explicit citation. Suffix and other components are excluded only by
    # their role-specific producer evidence above.
    return _prefix_key(rows)


def _prefix_key(rows):
    if not rows:
        return (1,)
    counts = tuple(-sum(number == value for number, _ in rows)
                   for value in (6, 5, 4))
    by_number = tuple(tuple(sorted(locant for number, locant in rows
                                   if number == value))
                      for value in (6, 5, 4))
    return (0, *counts, *by_number)
